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

"""
轨道计算
1. 设定动力学模型。
2. 对运动方程数值积分计算轨道。
3. 对变分方程数值积分计算状态转移矩阵和敏感矩阵。
"""

ld = logger.debug
lw = logger.warning
le = logger.error

__all__ = [
    'CoupleIntegrator'
]

class CoupleIntegrator(Integrator):
    '''
    这是交互作用积分函数
    主要应用与天体和卫星的共同高精度积分
    海王星和海卫一
    '''
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
        
        self.orbiter_pri = orbiter_primary            #主天体
        self.orbiter_sec = orbiter_secondary          #卫星

        #force_models = [forces_nep,force_nep_by_tri, forces_tri,force_tri_by_nep]   #动力学模型分配
        self.forces_nep = forces_primary_secondary[0]
        self.force_nep_by_tri = forces_primary_secondary[1] 
        self.forces_tri = forces_primary_secondary[2]
        self.force_tri_by_nep = forces_primary_secondary[3]

        #self.orbiter.set_integrator(self)             
        self.orbiter = orbiter_primary
        self.orbiter.set_integrator(self)
        
        # 时间定义
        # 前后各多增加一小时，避免生成观测值超出轨道弧段
        self.t_start = start_time - extended_seconds  # start of the arc
        self.t_end = end_time + extended_seconds  # end of the arc
        self.step = step
       
        if t_eval is not None:
            self.t_eval = t_eval
        else:
            self.t_eval = np.arange(start_time, end_time + .1 * self.step, self.step)

        self.t_ref = t_ref
        y_read_ref = np.zeros(12)
        y_read_ref[:6] = y_ref_pri
        y_read_ref[6:] = y_ref_sec
        self.y_ref = y_read_ref

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

        # interpolator of state vectors
        self.state_result_interpolator = None
        # interpolator of state transition matrix, sensitivity matrix
        self.var_result_interpolator = None

    # region Equation of motion
    def integrate_orbit(self):
        t_ref = self.t_ref
        y_ref = self.y_ref
        # backward integration
        res_backward = self._integrate_one_direction(t_ref, self.t_start, y_ref)
        # forward integration
        res_forward = self._integrate_one_direction(t_ref, self.t_end, y_ref)

        self.state_result_interpolator = self.combine_two_way_arcs(res_backward, res_forward)

        return res_backward, res_forward

    def _integrate_one_direction(self, t0, tn, y0):
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
        "改：协同积分方法改变"
        sol = solve_ivp(
            self._coupled_equation_of_motion, (t0, tn), y0, dense_output=True,
            method=self.equation_of_motion_method, rtol=self.rel_err, atol=self.abs_err,
            **self.integrator_kwargs
        )
        assert isinstance(sol, OptimizeResult)
        return sol

    def _coupled_equation_of_motion(self, t, y_nep_tri):
        "改：写协同积分的变分方程"
        #a_forces = np.array([f.calc_acceleration(t, y) for f in self.force_models])
        #a_sum = np.sum(a_forces, axis=0)
        #y_dot = np.array([*y[3:], *a_sum])
        #return y_dot
        y_nep = y_nep_tri[:6]
        y_tri = y_nep_tri[6:]
        
        #海王星
        a_nep_forces = np.array([f.calc_acceleration(t, y_nep) for f in self.forces_nep])
        a_nep_forces = np.append(a_nep_forces, [self.force_nep_by_tri.calc_acceleration(t, y_tri)],axis=0)
        a_nep_sum = np.sum(a_nep_forces, axis=0)

        y_nep_dot = np.array([*y_nep[3:], *a_nep_sum])

        #海卫一
        a_tri_forces = np.array([f.calc_acceleration(t, y_tri , y_nep) for f in self.forces_tri])
        a_tri_forces = np.append(a_tri_forces, np.array([f.calc_acceleration(t, y_tri) for f in self.force_tri_by_nep]) ,axis=0)
        a_tri_sum = np.sum(a_tri_forces, axis=0)

        y_tri_dot = np.array([*y_tri[3:], *a_tri_sum])

        y_dot_all = np.concatenate([y_nep_dot, y_tri_dot])
        return y_dot_all

    def interpolate_equation_of_motion_results(self, ts):
        "改：取主天体轨道"
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
        print("用到了")
        return self.interpolate_equation_of_motion_results(ts)[:, :3]

    # endregion

    # region Variational equation
    def integrate_orbit_with_partial(self, pm_manager):
        d = sum(pm_manager.get_param_attr_list(pm_manager[self], 'pm_num'))
        tm = self.t_ref
        ym = self.y_ref
        
        Y0 = np.zeros((12, d + 1))
        Y0[:, 0] = ym
        Y0[:, 1:13] = np.eye(12)
        Y0[:, 13:] = 0

        res_backward = self._integrate_one_direction_with_partial(tm, self.t_start, Y0, pm_manager)
        res_forward = self._integrate_one_direction_with_partial(tm, self.t_end, Y0, pm_manager)

        self.var_result_interpolator = self.combine_two_way_arcs(res_backward, res_forward)
        return res_backward, res_forward

    def _integrate_one_direction_with_partial(self, t0, tn, Y0, pm_manager):
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
        sol = solve_ivp(
            self._coupled_variational_equation, (t0, tn), Y0.ravel(), dense_output=True,
            method=self.variational_equation_method, rtol=self.rel_partial_err, atol=self.abs_partial_err,
            args=(pm_manager,),
            **self.integrator_kwargs
        )
        assert isinstance(sol, OptimizeResult)
        return sol

    # from .parameter_models import ParametersManager
    def _coupled_variational_equation(self, t, Y, pm_manager):
        '''重新定义变分方程12*12'''
        # reshape Y to (12, 1 + d)
        Y_mat = Y.reshape((12, -1))
        y = Y_mat[:, 0]
        phi = Y_mat[:, 1:13]
        s = Y_mat[:, 13:]

        dy_dt = self._coupled_equation_of_motion(t, y) # 1*12
        
        # dv_dr = 0
        df_dy = np.zeros((12, 12))

        # dv_dv = I & dv_dv = I
        df_dy[:3, 3:6] = np.eye(3)
        df_dy[6:9, 9:12] = np.eye(3)
        
        # Neptune
        # da_dr
        da_nep_dr_nep = np.array([f.calc_da_dr(t, y[:6]) for f in self.forces_nep])
        da_nep_dr_tri = np.array([self.force_nep_by_tri.calc_da_dr_tri(t, y[6:])])
        df_dy[3:6, :3] = np.sum(da_nep_dr_nep, axis=0)
        df_dy[3:6, 6:9] = np.sum(da_nep_dr_tri, axis=0)

        # da_dv
        # 均为0

        # Triton
        # da_dr
        da_tri_dr_tri = np.array([f.calc_da_dr(t, y[6:], y[:6]) for f in self.forces_tri])
        da_tri_dr_tri = np.append(da_tri_dr_tri, np.array([f.calc_da_dr(t, y[6:]) for f in self.force_tri_by_nep]) ,axis=0)
        da_tri_dr_nep = np.array([f.calc_da_dr_nep(t, y[6:], y[:6]) for f in self.forces_tri])
        df_dy[9:, 6:9] = np.sum(da_tri_dr_tri, axis=0)
        df_dy[9:, :3] = np.sum(da_tri_dr_nep, axis=0)

        
        # da_dv
        # 潮汐
        # da_tri_dv_tri = np.array([self.force_tri_by_nep[1].calc_da_dv(t, y[6:])])
        # df_dy[9:, 9:] = np.sum(da_tri_dv_tri, axis=0)

        Y_dot = np.zeros(Y_mat.shape)
        Y_dot[:, 0] = dy_dt
        Y_dot[:, 1:13] = df_dy @ phi
        
        dynamical_list = pm_manager[self]
        if len(dynamical_list) > 1:  # which means not only the initial state vector
            da_dp_dynamic_list_1 = []
            # da_dp_dynamic_list_2 = []
            da_dp_dynamic_list_3 = []
            # da_dp_dynamic_list_4 =  []
            # iterate over parameters after the initial state vector.
            # for param in dynamical_list[1:]:
            # 单情况
                # da_dp_func = param.get_da_dp_function()
                # da_dp_dynamic_list_1.append(da_dp_func(t, y[6:]))
            # 多情况
            # 1.0 重力场参数
            param_grav = dynamical_list[1]
            da_dp_func_1 =  param_grav.get_da_dp_function()
            da_dp_dynamic_list_1.append(da_dp_func_1(t, y[6:]))
            # 2.0 潮汐参数
            # param_tf = dynamical_list[2]
            # da_dp_func_2 =  param_tf.get_da_dp_function()
            # da_dp_dynamic_list_2.append(da_dp_func_2(t, y[6:]))
            # 3.0 海卫一重力参数
            param_gm = dynamical_list[2]
            da_dp_func_3 =  param_gm.get_da_dp_function()
            da_dp_dynamic_list_3.append(da_dp_func_3(t, y[6:]))
            # 4.0 海王星定向参数
            # param_or = dynamical_list[3]
            # da_dp_func_4 =  param_or.get_da_dp_function()
            # da_dp_dynamic_list_4.append(da_dp_func_4(t, y[6:]))
            
            da_dp_dynamical_1 = np.column_stack(da_dp_dynamic_list_1)
            #da_dp_dynamical_2 = np.column_stack(da_dp_dynamic_list_2)
            da_dp_dynamical_3 = np.column_stack(da_dp_dynamic_list_3)
            #da_dp_dynamical_4 = np.column_stack(da_dp_dynamic_list_4)

            # 海卫一的偏导数
            vstack_gr = np.vstack([np.zeros_like(da_dp_dynamical_1), np.zeros_like(da_dp_dynamical_1), np.zeros_like(da_dp_dynamical_1), da_dp_dynamical_1])
            # vstack_tf = np.vstack([np.zeros_like(da_dp_dynamical_2), np.zeros_like(da_dp_dynamical_2), np.zeros_like(da_dp_dynamical_2), da_dp_dynamical_2])
            vstack_gm = np.vstack([np.zeros_like(da_dp_dynamical_3), da_dp_dynamical_3, np.zeros_like(da_dp_dynamical_3), np.zeros_like(da_dp_dynamical_3)])
            # vstack_or = np.vstack([np.zeros_like(da_dp_dynamical_4), np.zeros_like(da_dp_dynamical_4), np.zeros_like(da_dp_dynamical_4), da_dp_dynamical_4])

            df_dy_s = df_dy @ s
            df_dy_s[: , :3] += vstack_gr 
            # df_dy_s[: , 3:4] += vstack_tf
            df_dy_s[: , 3:4] += vstack_gm
            # df_dy_s[: , 4:6] += vstack_or

            Y_dot[:, 13:] = df_dy_s

        Y_dot_flatten = Y_dot.ravel()
        return Y_dot_flatten

    def interpolate_variational_equation_results(self, ts):
        """
        Interpolate the variational equation results, including the state, state transition matrix
        and the sensitivity matrix.

        Parameters
        ----------
        ts: ndarray of shape (n,)
            times to interpolate.
        Returns
        -------
        ys: ndarray of shape (n, 6)
            state vectors at ts.
        dy_dp_dynamical: ndarray of shape (n, 6, d)
            state transition matrices and sensitivity matrices at ts.
        """
        if not isinstance(ts, np.ndarray):
            ts = np.array(ts)
        # 重写一下矩阵大小以及取其中海王星状态部分
        self.validate_interpolate_time(ts)
        YS = self.var_result_interpolator(ts).T.reshape((ts.shape[0], 12, -1))
        ys = YS[:, :, 0]
        dy_dp_dynamical = YS[:, :, 1:]

        return ys, dy_dp_dynamical

    # endregion

    @staticmethod
    def combine_two_way_arcs(sol_backward, sol_forward):
        """
        Parameters
        ----------
        sol_backward : OdeSolution
            arc1, backward integration, from tm to t0.
        sol_forward : OdeSolution
            arc2, forward integration, from t0 to tn.

        Returns
        -------
        """
        t1, i1 = sol_backward.ts, sol_backward.interpolants
        t2, i2 = sol_forward.ts, sol_forward.interpolants

        t3 = np.concatenate([t1[::-1], t2[1:]])
        i3 = np.concatenate([i1[::-1], i2])
        sol_combine = OdeSolution(t3, i3)
        return sol_combine

    @staticmethod
    def combine_arcs(sols: List[OdeSolution]):
        ts = np.concatenate([
            sols[0].ts,
            *[sol.ts[1:] for sol in sols[1:]]
        ])
        interpolants = np.concatenate([sol.interpolants for sol in sols])
        sol_combine = OdeSolution(ts, interpolants)
        return sol_combine

    def validate_interpolate_time(self, ts):
        if any(ts > self.t_end) or any(ts < self.t_start):
            raise Exception(
                f"Orbit can interpolate from {get_date_of_tdb(self.t_start)} to {get_date_of_tdb(self.t_end)}. "
                f"While got ts from {get_date_of_tdb(ts[0])} to {get_date_of_tdb(ts[-1])}."
            )

    def split_time_span_with_empirical_ts(self, t0, tn):
        # split the time span according to the empirical acceleration times.
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

#定义天体积分的类

# class Dnbody():
#     def __init__(self):
#         # 常量定义（需根据实际情况调整）
#         MaxBody = 2        # 最大天体数
#         NumPlanet = 10     # 行星数量（示例值）
#         NumPhi = 6         # 变分矩阵维度
#         VC2 = 29979.0638238976 
#         GM = 2.9591220828411956e-04
#         PI = 3.141592653589793
#         au = 149597870.700
#         arcsecond = PI/3600/180 
#         tc_au  = 173.14463267424
#         MJD = 2400000.5
#         EOPMax = 50000
#         numb = 1           # 当前天体数量

#         # 时间参数
#         self.timebegin = 0.0
#         self.time = 0.0
#         self.timestep = 0.0
#         self.timeend = 0.0
#         self.toutput = 0          # 输出时间标记
        
#         # 状态向量 [MaxBody][12]
#         self.xv = np.zeros((MaxBody, 12), dtype=np.float64)

#         # 天体基础数据
#         self.ID = np.zeros(MaxBody, dtype=np.int32)      # 天体ID
        
#         # 行星系统参数
#         self.pmass = np.zeros(NumPlanet, dtype=np.float64)     # 行星质量
#         self.pxv = np.zeros((NumPlanet, 6), dtype=np.float64)  # 行星状态
        
#         # 变分方程矩阵
#         self.phi = np.zeros((MaxBody, NumPhi, NumPhi), dtype=np.float64)
#         self.phidec = np.zeros((MaxBody, NumPhi, NumPhi), dtype=np.float64)
        
#         # 小行星带参数
#         self.asmass = np.zeros(400, dtype=np.float64)
#         self.aspos = np.zeros((400, 3), dtype=np.float64)
#         self.asname = np.zeros(400, dtype=np.int32)
#         self.asnum = 0
        
#         # 太阳参数
#         self.Spole = np.zeros(3, dtype=np.float64)  # 极轴方向
#         self.SJ2 = 0.0               # 太阳J2项
#         self.SRad = 0.0              # 太阳半径
        
#         # 火星卫星参数
#         self.marsmass = np.zeros(5, dtype=np.float64)
#         self.marspos = np.zeros((5, 3), dtype=np.float64)
#         self.marsname = np.zeros(5, dtype=np.int32)
#         self.marsnum = 0
        
#         # KBO参数
#         self.kbosmass = np.zeros(40, dtype=np.float64)
#         self.kbospos = np.zeros((40, 3), dtype=np.float64)
#         self.kbosname = np.zeros(40, dtype=np.int32)
#         self.kbosnum = 0
        
#         # 重力场系数
#         self.GC = np.zeros((3, 5, 5), dtype=np.float64)  # 重力场余弦项
#         self.GS = np.zeros((3, 5, 5), dtype=np.float64)  # 重力场正弦项
#         self.Rad = np.zeros(4, dtype=np.float64)         # 天体半径
        
#         # 极轴方向矩阵
#         self.PD = np.zeros((3, 3), dtype=np.float64)
        
#         # 坐标转换矩阵
#         self.rc2t = np.zeros((3, 3), dtype=np.float64)    # 旋转矩阵
#         self.t2rc = np.zeros((3, 3), dtype=np.float64)    # 逆旋转矩阵
        
#         # 地球参数
#         self.Ek2 = np.zeros(3, dtype=np.float64)
#         self.Etau = np.zeros(3, dtype=np.float64)
        
#         # 月球参数
#         self.MJ2 = 0.0                                  # 月球J2项
#         self.Mbeta = 0.0                                # 月球beta参数
#         self.Mgamma = 0.0                               # 月球gamma参数
#         self.MI = np.zeros((3, 3), dtype=np.float64)    # 惯性张量
#         self.MdI = np.zeros((3, 3), dtype=np.float64)   # 惯性张量导数
#         self.MIt = np.zeros((3, 3), dtype=np.float64)   # 转置惯性张量
#         self.Mtau = 0.0                                 # 月球时间参数
#         self.Mk2 = 0.0                                  # 月球k2参数
        
#         # 质量参数
#         self.mass = np.zeros(4, dtype=np.float64)
#         self.EMrat = 0.0              # 地球/质量比
        
#         # 地球定向参数
#         self.eop = np.zeros((EOPMax, 3), dtype=np.float64)
#         self.EOPjd = 0.0
#         self.EOPnum = 0
        
#         # KBO参考点
#         self.kboRpos = np.zeros((36, 3), dtype=np.float64)
#         self.kboRmass = 0.0

# class NewIntegrator(Integrator):
#     """BS算法内部状态容器"""
#     def _initialize_attributes(self):
#         """初始化属性"""
      
#         # 太阳参数
#         self.Spole = np.zeros(3, dtype=np.float64)
#         self.SJ2 = 2.1961391516529825E-07
#         self.SRad = 6.96e5 / self.AU
        
#         # 月球参数
#         self.MJ2 = 0.000204312007
#         self.Mbeta = 0.0006316121
#         self.Mgamma = 0.0002278583

#     def _load_all_data(self):
#         """执行完整初始化流程"""
#         self._load_initial_conditions()
#         self._setup_physical_parameters()
#         self._initialize_phi_matrix()
#         self._load_asteroid_data()
#         self._load_mars_satellites()
#         self._setup_gravity_coefficients()

#     def _load_initial_conditions(self):
#         """加载初始状态文件"""
#         input_file = self.data_path / 'initial_conditions.txt'
#         try:
#             with open(input_file, 'r') as f:
#                 # 假设文件格式：时间 + 6个状态量
#                 data = np.loadtxt(f)
#                 self.time = data[0]
#                 self.xv[0, :6] = data[1:7]
#         except Exception as e:
#             raise RuntimeError(f"Failed to load initial conditions: {str(e)}")

#     def _setup_physical_parameters(self):
#         """配置物理常数"""
#         # 行星质量 (DE430值)
#         self.pmass = np.array([
#             2.9591309705483544E-04,  # Sun
#             4.9125001948893182E-11,  # Mercury
#             7.2434523326441187E-10,  # Venus
#             8.8876924467071022e-10,  # Earth
#             2.8253458252257917E-07,  # Mars
#             8.4597059933762903E-08,  # Jupiter
#             1.2920265649682399E-08,  # Saturn
#             1.5243573478851939E-08,  # Uranus
#             2.1750964648933581E-12,  # Neptune
#             1.0931894624024351e-11   # Moon
#         ])

#         self.mass = np.array([
#             2.9591309705483544E-04,  # Sun
#             8.8876924467071022E-10,  # Earth
#             1.0931894624024351e-11,  # Moon 
#             2.8253458252257917E-07,  # Mars
#         ])
#         self.EMrat = 8.1300568221497215E+01; # 地月质量比
#         # 太阳极轴方向 (转换为弧度)
#         self.Spole = np.deg2rad([286.13, 63.87])

#     def _initialize_phi_matrix(self):
#         """初始化变分矩阵"""
#         for i in range(self.numb):
#             self.phi[i] = np.eye(self.NUM_PHI)
#             self.phidec[i] = np.zeros((self.NUM_PHI, self.NUM_PHI))

#     def _load_asteroid_data(self):
#         """加载小行星数据"""
#         as_file = self.data_path / 'asteroids.dat'
#         try:
#             data = np.loadtxt(as_file, dtype=np.int32)
#             self.asnum = data.shape[0]
#             self.asname[:self.asnum] = data[:, 0]
#             self.asmass[:self.asnum] = data[:, 1]
#         except Exception as e:
#             print(f"Warning: Asteroid data not loaded - {str(e)}")

#     def _load_mars_satellites(self):
#         """加载火星卫星数据"""
#         mars_file = self.data_path / 'mars_satellites.dat'
#         try:
#             data = np.loadtxt(mars_file, dtype=np.int32)
#             self.marsnum = data.shape[0]
#             self.marsname[:self.marsnum] = data[:, 0]
#             self.marsmass[:self.marsnum] = data[:, 1]
#         except Exception as e:
#             print(f"Warning: Mars satellite data not loaded - {str(e)}")

#     def _setup_gravity_coefficients(self):
#         """配置非球形引力系数"""

#         self.SJ2 = 2.1961391516529825E-07
#         self.SRad = 6.9600000000000000E+05 / au

#         # 太阳J2项
#         self.GC[0, 2, 0] = 2.1961391516529825E-07
 
#         # 地球引力场 (EGM2008)
#         self.GC[1,2,:3] = [1.0826261738522e-3, -2.6673947523748e-10, 1.5746153257229e-06]
#         self.GC[1,3,:4] = [-2.5324105185677e-06, 2.1931496313133e-06, 3.0904390039165e-07, 1.0058351340882e-07]
#         self.GC[1,4,:5] = [-1.619897599917e-06, -5.0864356043958e-07, 7.8374545740455e-08, 5.9215017763967e-08, -3.9832042487319e-09]
    
#         self.GS[1,2,1:3] = [1.787270648524e-09, -9.0387278919657e-07]
#         self.GS[1,3,1:4] = [2.680870894009e-07, -2.1143062093348e-07, 1.9722158183572e-07]
#         self.GS[1,4,1:5] = [-4.4926543214381e-07, 1.4813503724886e-07, -1.2009461262296e-08, 6.5246728718769e-09]

#         # 月球引力场 -------------------------------------------------
#         self.GC[2,3,:4] = [-8.45970269745946e-06, 2.84807411955929e-05, 4.84494206197706e-06, 1.67561781341146e-06]
#         self.GS[2,3,1:4] = [5.89155515553186e-06, 1.68447439627839e-06, -2.47427143798058e-07]

#         # 天体物理参数 -----------------------------------------------
#         self.Rad = np.array([696000, 6378.1366, 1738, 0, 3389.5]) / self.AU  # 索引4对应Mars
#         self.MJ2, self.Mbeta, self.Mgamma = 0.000204312007, 0.0006316121, 0.0002278583
#         self.Mtau, self.Mk2 = 0.1667165558, 0.0299221167

# class DBS:
#     def __init__(self, numb, bs_max, num_phi):
#             # Tkk结构: [numb][bs_max][6]
#             self.Tkk = np.zeros((numb, bs_max, 6), dtype=np.float64)
#             # 中间变量m/mphi
#             self.m = np.zeros((numb, bs_max+2, 6), dtype=np.float64)
#             self.dt = 0.0
#             self._initialize_attributes()
#             self._load_all_data()


# #下面是积分的代码部分
# #---------------- 调用BS积分程序 -------------------
# def detneo(pnbody):
#     flag = BS(pnbody)
#     if flag == 1:
#         pnbody.timestep /= 10
#         for i in range(10):
#             flag = BS(pnbody)
#             if flag == 1:
#                 print("error BS!")
#             pnbody.time += pnbody.timestep
#         pnbody.timestep *= 10
#     else:
#         pnbody.time += pnbody.timestep

# #---------------- Bulirsch-Stoer积分核心 ------------
# def BS(pnbody):
 
#     Accuracy=1e-12
#     BSMax=200

#     y = Dnbody(pnbody.numb)
#     bs = DBS(pnbody.numb)
    
#     for num in range(4):
#         BSmidpoint(pnbody, bs, num)
#         BSrecurrence(pnbody, bs, num)
    
#     for num in range(4, BSMax):
#         BSmidpoint(pnbody, bs, num)
#         BSrecurrence(pnbody, bs, num)
        
#         error = 0.0
#         errorphi = 0.0
        
#         # 计算误差
#         for i in range(pnbody.numb):
#             error = max(error, np.max(np.abs(bs.Tkk[i, 0, :])))
#             #errorphi = max(errorphi, np.max(np.abs(bs.Tkkphi[i, :, :, 0])))
        
#         if error < Accuracy:
#             # 更新状态
#             for i in range(pnbody.numb):
#                 # 更新xv
#                 y.xv[i, :6] = bs.Tkk[i, 0, :]
#                 y.xv[i, 6:] = 0.0
#                 for numk in range(1, num+1):
#                     y.xv[i, :6] += bs.Tkk[i, numk, :]
            
#             CS(pnbody, y)
#             return 0
    
#     print(f"Wrong triton_BS! Iterations: {num}")
#     return 1

# #---------------- 递推法 ----------------------
# def BSrecurrence(pnbody, pbs, num):
#     num += 1
#     BSMax=200

#     h2 = np.zeros(BSMax)
#     for numk in range(1, num+1):
#         h2[numk-1] = 0.25 / (numk**2)
    
#     for numk in range(num-1, 0, -1):
#         tmp0 = 1.0 / (h2[numk-1] - h2[num-1])
#         tmp1 = tmp0 * h2[numk]
#         tmp2 = tmp0 * h2[num-1]
        
#         # 更新Tkk
#         for i in range(pnbody.numb):
#             pbs.Tkk[i, numk-1, :] = tmp1 * pbs.Tkk[i, numk, :] - tmp2 * pbs.Tkk[i, numk-1, :]
          
# #---------------- 中点法 ----------------------
# def BSmidpoint(pnbody, pbs, num):
#     num += 1
#     n = 2 * num
#     dt = pnbody.timestep
#     h = dt / n
#     h2 = 2 * h
#     pbs.dt = h
    
#     # 初始化中间变量
#     pbs.m[:, 0, :] = 0.0

    
#     # 需实现BSf函数!!!
#     BSf(pnbody, pbs, 0, 0, h, 1, 0.0) 
    
#     for m in range(1, n):
#         BSf(pnbody, pbs, m-1, m, h2, m+1, m*h)
    
#     BSf(pnbody, pbs, n-1, n, h, n+1, dt)
    
#     # 计算Tkk和Tkkphi
#     for i in range(pnbody.numb):
#         pbs.Tkk[i, num-1, :] = (pbs.m[i, n, :] + pbs.m[i, n+1, :]) / 2


# #---------------- 补偿求和 ---------------------
# def CS(pnbody, det):
#     y = Dnbody(pnbody.numb)
#     for i in range(pnbody.numb):
#         # 更新xv
#         y.xv[i, 6:] = pnbody.xv[i, 6:] + det.xv[i, :6]
#         y.xv[i, :6] = pnbody.xv[i, :6] + y.xv[i, 6:]
#         y.xv[i, 6:] += pnbody.xv[i, :6] - y.xv[i, :6]
#         pnbody.xv[i] = y.xv[i].copy()
       

# # TODO: 需补充BSf函数实现 (动力学方程)
# def BSf(pnbody, pbs, m_prev, m_curr, h, m_next, t):
    """
    需根据具体问题实现:
    - 计算天体的加速度和变分方程
    - 更新pbs.m和pbs.mphi中的中间值
    """
    pass 