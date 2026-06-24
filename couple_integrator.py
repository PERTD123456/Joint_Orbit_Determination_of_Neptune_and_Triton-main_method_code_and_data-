"""Coupled numerical integration for a primary planet and its satellite (just for one now).

Author: Yixiao Liu, Wutong Gao
Affiliation: Wuhan University, Wuhan, China
Email: lyx2002@whu.edu.cn
Last updated: 2024-06-20

Description:
This module provides tools for simultaneously integrating the equations of
motion of two dynamically coupled bodies. It also supports integration of the
associated variational equations to obtain state-transition and sensitivity
matrices. 

The implementation is used for the coupled Neptune--Triton dynamical model,
but the class interface is sufficiently general for other primary--satellite
systems with compatible force-model implementations.
"""


from typing import List
import numpy as np
from more_itertools import pairwise
from scipy.integrate import solve_ivp, OdeSolution
from scipy.optimize import OptimizeResult
from .force_models import EmpiricalAcceleration
from .global_object import *
from .helpers.spice_helper import get_date_of_tdb
from .pod_logger import logger
from .time import PODTime
from .integrator import *

ld = logger.debug
lw = logger.warning
le = logger.error

__all__ = ['CoupleIntegrator']

class CoupleIntegrator(Integrator):
    """Numerically integrate the coupled motion of a primary and a satellite."""

    # 1. Initialization (Integrator is a simple base class for integrating for orbit propagation)
    def __init__(self, orbiter_primary, orbiter_secondary, forces_primary_secondary,
                 start_time,
                 end_time,
                 step,
                 t_ref,
                 y_ref_pri,
                 y_ref_sec,
                 t_eval=None,
                 method='DOP853',
                 abs_err=1e-11,
                 rel_err=1e-11,
                 abs_partial_err=1e-9,
                 rel_partial_err=1e-9,
                 extended_seconds=3600,
                 integrator_kwargs=None,
                 ):
        
        # 1.1 Orbiter objects
        self.orbiter_pri = orbiter_primary
        self.orbiter_sec = orbiter_secondary
        self.orbiter = orbiter_primary
        self.orbiter.set_integrator(self)

        # 1.2 Force-model configuration
        #force_models = [forces_nep,force_nep_by_tri, forces_tri,force_tri_by_nep]
        # [0] forces acting on Neptune
        # [1] acceleration of Neptune caused by Triton
        # [3] additional forces acting on Triton
        # [2] forces acting on Triton that depend on Neptune'
        self.forces_nep = forces_primary_secondary[0]
        self.force_nep_by_tri = forces_primary_secondary[1] 
        self.forces_tri = forces_primary_secondary[2]
        self.force_tri_by_nep = forces_primary_secondary[3]

        # 1.3 Integration interval
        # Extend both ends of the nominal arc to avoid interpolation failures 
        # for observations located close to the arc boundaries.
        self.t_start = start_time - extended_seconds
        self.t_end = end_time + extended_seconds
        self.step = step
       
        if t_eval is not None:
            self.t_eval = t_eval
        else:
            self.t_eval = np.arange(start_time, end_time + .1 * self.step, self.step)
        
        # 1.4 Reference epoch and initial states
        self.t_ref = t_ref
        y_read_ref = np.zeros(12)
        y_read_ref[:6] = y_ref_pri
        y_read_ref[6:] = y_ref_sec
        self.y_ref = y_read_ref

        # 1.5 Numerical integration settings
        # Separate integration methods may be specified for the state equations 
        # and the variational equations.
        if isinstance(method, list):
            self.equation_of_motion_method = method[0]
            self.variational_equation_method = method[1]
        else:
            self.equation_of_motion_method = self.variational_equation_method = method

        self.abs_err = abs_err
        self.rel_err = rel_err

        self.abs_partial_err = abs_partial_err
        self.rel_partial_err = rel_partial_err

        if integrator_kwargs is None:
            integrator_kwargs = {}
        self.integrator_kwargs = integrator_kwargs

        # 1.6 Dense-output interpolators
        # interpolator of state vectors
        self.state_result_interpolator = None
        # interpolator of state transition matrix, sensitivity matrix
        self.var_result_interpolator = None

    # 2. Coupled orbit integration
    def integrate_orbit(self):
        """Integrate the coupled state equations backward and forward."""
        t_ref = self.t_ref
        y_ref = self.y_ref
        # backward integration
        res_backward = self._integrate_one_direction(t_ref, self.t_start, y_ref)
        # forward integration
        res_forward = self._integrate_one_direction(t_ref, self.t_end, y_ref)

        self.state_result_interpolator = self.combine_two_way_arcs(res_backward, res_forward)

        return res_backward, res_forward

    def _integrate_one_direction(self, t0, tn, y0):
        """Integrate the state equations in one temporal direction."""
        sol_list = []

        t_nodes = self.split_time_span_with_empirical_ts(t0, tn)
        yi = y0
        for t1, t2 in pairwise(t_nodes):
            ode_res_i = self._integrate_one_direction_continuous_arc(t1, t2, yi)
            if not ode_res_i.success:
                lw(ode_res_i.message)

            sol_i = ode_res_i.sol  # type:OdeSolution
            sol_list.append(sol_i)
            yi = sol_i(t2)

            ld(f"{get_date_of_tdb(t1)}, {get_date_of_tdb(t2)} arc integration finished")

        sol = self.combine_arcs(sol_list)
        return sol

    def _integrate_one_direction_continuous_arc(self, t0, tn, y0):
        """Integrate one continuous arc of the coupled state equations."""
        sol = solve_ivp(
            self._coupled_equation_of_motion, (t0, tn), y0, dense_output=True,
            method=self.equation_of_motion_method, rtol=self.rel_err, atol=self.abs_err,
            **self.integrator_kwargs
        )
        assert isinstance(sol, OptimizeResult)
        return sol
    
    # 3. Coupled equations of motion
    def _coupled_equation_of_motion(self, t, y_nep_tri):
        """Evaluate the coupled state derivatives of Neptune and Triton."""
        # 3.1 Separate the two state vectors
        y_nep = y_nep_tri[:6]
        y_tri = y_nep_tri[6:]
        
        # 3.2 Neptune acceleration
        a_nep_forces = np.array([f.calc_acceleration(t, y_nep) for f in self.forces_nep])
        a_nep_forces = np.append(a_nep_forces, [self.force_nep_by_tri.calc_acceleration(t, y_tri)],axis=0)
        a_nep_sum = np.sum(a_nep_forces, axis=0)

        y_nep_dot = np.array([*y_nep[3:], *a_nep_sum])

        # 3.3 Triton acceleration
        a_tri_forces = np.array([f.calc_acceleration(t, y_tri , y_nep) for f in self.forces_tri])
        a_tri_forces = np.append(a_tri_forces, np.array([f.calc_acceleration(t, y_tri) for f in self.force_tri_by_nep]) ,axis=0)
        a_tri_sum = np.sum(a_tri_forces, axis=0)

        y_tri_dot = np.array([*y_tri[3:], *a_tri_sum])

        # 3.4 Return the complete 12-dimensional state derivative
        y_dot_all = np.concatenate([y_nep_dot, y_tri_dot])
        return y_dot_all

    def interpolate_equation_of_motion_results(self, ts):
        """Interpolate the propagated coupled state vectors."""
        if not isinstance(ts, np.ndarray):
            ts = np.array(ts)
        self.validate_interpolate_time(ts)
        if self.state_result_interpolator:
            ys = self.state_result_interpolator(ts).T
        elif self.var_result_interpolator:
            ys = self.var_result_interpolator(ts).T.reshape((ts.shape[0], 12, -1))[:, :, 0]
        else:
            raise Exception("No integration results to interpolate ts")
        ys = ys[:,:]
        return ys

    def interpolate_positions(self, ts):
        """Interpolate the Cartesian position of the primary body."""
        return self.interpolate_equation_of_motion_results(ts)[:, :3]


    # 5. Variational-equation integration
    def integrate_orbit_with_partial(self, pm_manager):
        """Integrate the states and their parameter partial derivatives."""
        d = sum(pm_manager.get_param_attr_list(pm_manager[self], 'pm_num'))
        tm = self.t_ref
        ym = self.y_ref

        # 5.1 Construct the initial augmented matrix
        Y0 = np.zeros((12, d + 1))
        Y0[:, 0] = ym
        Y0[:, 1:13] = np.eye(12)
        Y0[:, 13:] = 0

        # 5.2 Backward and forward integration
        res_backward = self._integrate_one_direction_with_partial(tm, self.t_start, Y0, pm_manager)
        res_forward = self._integrate_one_direction_with_partial(tm, self.t_end, Y0, pm_manager)

        self.var_result_interpolator = self.combine_two_way_arcs(res_backward, res_forward)
        return res_backward, res_forward

    def _integrate_one_direction_with_partial(self, t0, tn, Y0, pm_manager):
        """Integrate the variational equations in one direction."""
        sol_list = []
        t_nodes = self.split_time_span_with_empirical_ts(t0, tn)
        
        Yi = Y0
        for t1, t2 in pairwise(t_nodes):
            ode_res_i = self._integrate_one_direction_continuous_arc_with_partial(t1, t2, Yi, pm_manager)
            if not ode_res_i.success:
                lw(f"{PODTime.from_tdb_sec([t1, t2]).datetime} integrated failed.")
                lw(f"{PODTime.from_tdb_sec(ode_res_i.t[[0, -1]]).datetime} is valid interval.")
                lw(f"Integrator message: {ode_res_i.message}")
            sol_i = ode_res_i.sol  # type:OdeSolution
            sol_list.append(sol_i)
            Yi = sol_i(t2)

        sol = self.combine_arcs(sol_list)
        return sol

    def _integrate_one_direction_continuous_arc_with_partial(self, t0, tn, Y0, pm_manager):
        """Integrate one continuous arc of the variational equations."""
        sol = solve_ivp(
            self._coupled_variational_equation, (t0, tn), Y0.ravel(), dense_output=True,
            method=self.variational_equation_method, rtol=self.rel_partial_err, atol=self.abs_partial_err,
            args=(pm_manager,),
            **self.integrator_kwargs
        )
        assert isinstance(sol, OptimizeResult)
        return sol

    # 6. Coupled variational equations
    def _coupled_variational_equation(self, t, Y, pm_manager):
        """Evaluate the coupled 12-state variational equations."""
        
        # 6.1 Recover the augmented state matrix
        Y_mat = Y.reshape((12, -1))
        y = Y_mat[:, 0]
        phi = Y_mat[:, 1:13]
        s = Y_mat[:, 13:]

        # Nominal state derivative.
        dy_dt = self._coupled_equation_of_motion(t, y)

        # 6.2 Initialize the 12 x 12 system Jacobian
        # dv_dr = 0
        df_dy = np.zeros((12, 12))

        # dv_dv = I & dv_dv = I
        df_dy[:3, 3:6] = np.eye(3)
        df_dy[6:9, 9:12] = np.eye(3)
        
        # 6.3 Neptune acceleration partial derivatives
        # da_dr
        da_nep_dr_nep = np.array([f.calc_da_dr(t, y[:6]) for f in self.forces_nep])
        da_nep_dr_tri = np.array([self.force_nep_by_tri.calc_da_dr_tri(t, y[6:])])
        df_dy[3:6, :3] = np.sum(da_nep_dr_nep, axis=0)
        df_dy[3:6, 6:9] = np.sum(da_nep_dr_tri, axis=0)

        # da_dv

        # 6.4 Triton acceleration partial derivatives
        # da_dr
        da_tri_dr_tri = np.array([f.calc_da_dr(t, y[6:], y[:6]) for f in self.forces_tri])
        da_tri_dr_tri = np.append(da_tri_dr_tri, np.array([f.calc_da_dr(t, y[6:]) for f in self.force_tri_by_nep]) ,axis=0)
        da_tri_dr_nep = np.array([f.calc_da_dr_nep(t, y[6:], y[:6]) for f in self.forces_tri])
        df_dy[9:, 6:9] = np.sum(da_tri_dr_tri, axis=0)
        df_dy[9:, :3] = np.sum(da_tri_dr_nep, axis=0)

        # da_dv

        # 6.5 State-transition matrix propagation
        Y_dot = np.zeros(Y_mat.shape)
        Y_dot[:, 0] = dy_dt
        Y_dot[:, 1:13] = df_dy @ phi
        
        # 6.6 Dynamical-parameter sensitivity propagation
        dynamical_list = pm_manager[self]
        if len(dynamical_list) > 1: 
            # Current parameter order: 
            # [0] initial state parameters 
            # [1] Neptune gravity-field parameters
            # [2] Triton gravitational parameter 
            # [3] Neptune orientation parameters
            # Neptune gravity-field parameter partials
            
            da_dp_dynamic_list_1 = []
            da_dp_dynamic_list_2 = []
            da_dp_dynamic_list_3 = []

            # (1) Neptune gravity-field parameter partials
            param_grav = dynamical_list[1]
            da_dp_func_1 =  param_grav.get_da_dp_function()
            da_dp_dynamic_list_1.append(da_dp_func_1(t, y[6:]))
            # (2) Triton gravitational parameter partials
            param_gm = dynamical_list[2]
            da_dp_func_2 =  param_gm.get_da_dp_function()
            da_dp_dynamic_list_2.append(da_dp_func_2(t, y[6:]))
            # (3) Neptune orientation parameter partials
            param_or = dynamical_list[3]
            da_dp_func_3 =  param_or.get_da_dp_function()
            da_dp_dynamic_list_3.append(da_dp_func_3(t, y[6:]))
            
            da_dp_dynamical_1 = np.column_stack(da_dp_dynamic_list_1)
            da_dp_dynamical_2 = np.column_stack(da_dp_dynamic_list_2)
            da_dp_dynamical_3 = np.column_stack(da_dp_dynamic_list_3)

            vstack_gr = np.vstack([np.zeros_like(da_dp_dynamical_1), np.zeros_like(da_dp_dynamical_1), np.zeros_like(da_dp_dynamical_1), da_dp_dynamical_1])
            vstack_gm = np.vstack([np.zeros_like(da_dp_dynamical_2), da_dp_dynamical_2, np.zeros_like(da_dp_dynamical_2), np.zeros_like(da_dp_dynamical_2)])
            vstack_or = np.vstack([np.zeros_like(da_dp_dynamical_3), np.zeros_like(da_dp_dynamical_3), np.zeros_like(da_dp_dynamical_3), da_dp_dynamical_3])

            df_dy_s = df_dy @ s
            df_dy_s[: , :3] += vstack_gr 
            df_dy_s[: , 3:4] += vstack_gm
            df_dy_s[: , 4:6] += vstack_or

            Y_dot[:, 13:] = df_dy_s

        Y_dot_flatten = Y_dot.ravel()
        return Y_dot_flatten

    def interpolate_variational_equation_results(self, ts):
        """Interpolate states and variational-equation results."""
        if not isinstance(ts, np.ndarray):
            ts = np.array(ts)
        self.validate_interpolate_time(ts)
        YS = self.var_result_interpolator(ts).T.reshape((ts.shape[0], 12, -1))
        ys = YS[:, :, 0]
        dy_dp_dynamical = YS[:, :, 1:]

        return ys, dy_dp_dynamical

    # endregion

    @staticmethod
    def combine_two_way_arcs(sol_backward, sol_forward):
        """Combine backward and forward integration solutions."""
        t1, i1 = sol_backward.ts, sol_backward.interpolants
        t2, i2 = sol_forward.ts, sol_forward.interpolants

        t3 = np.concatenate([t1[::-1], t2[1:]])
        i3 = np.concatenate([i1[::-1], i2])
        sol_combine = OdeSolution(t3, i3)
        return sol_combine

    @staticmethod
    def combine_arcs(sols: List[OdeSolution]):
        """Combine consecutive continuous integration arcs."""
        ts = np.concatenate([
            sols[0].ts,
            *[sol.ts[1:] for sol in sols[1:]]
        ])
        interpolants = np.concatenate([sol.interpolants for sol in sols])
        sol_combine = OdeSolution(ts, interpolants)
        return sol_combine
    
    # 9. Time validation and arc splitting
    def validate_interpolate_time(self, ts):
        """Check whether the requested epochs lie inside the integrated arc."""
        if any(ts > self.t_end) or any(ts < self.t_start):
            raise Exception(
                f"Orbit can interpolate from {get_date_of_tdb(self.t_start)} to {get_date_of_tdb(self.t_end)}. "
                f"While got ts from {get_date_of_tdb(ts[0])} to {get_date_of_tdb(ts[-1])}."
            )

    def split_time_span_with_empirical_ts(self, t0, tn):
        """Split an integration interval at empirical-acceleration boundaries."""
        is_forward = t0 < tn

        if is_forward:
            t_min = t0
            t_max = tn
        else:
            t_min = tn
            t_max = t0

        t_nodes = [t_min]
        for f in [self.forces_nep, self.force_nep_by_tri, self.forces_tri, self.force_tri_by_nep]:
            if isinstance(f, EmpiricalAcceleration):
                # check time span
                for ti in [f.t0, f.tn]:
                    cond = t_min < ti < t_max
                    if cond:
                        t_nodes.append(ti)
        t_nodes.append(t_max)

        if not is_forward:
            t_nodes = t_nodes[::-1]
        assert is_sorted(t_nodes, ascending=is_forward)
        return t_nodes