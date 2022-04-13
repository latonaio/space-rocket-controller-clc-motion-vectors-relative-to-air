# TO-DO: Containerize to microservice
import numpy as np

# This microservice computes the motion vectors of
# the flight body relative to the air.
# Requires: Ground speeds, wind speeds
# Required by: Aerodynamic model, engine model
# Outputs: omitted for brevity
class AirSpeeds():
  def __init__(self, metadata_distributor):
    self.metadata_distributor = metadata_distributor
    self.constants = self.metadata_distributor.constants
    self.xyz_cg = self.metadata_distributor.get_var('xyz_cg')
    self.uvw_g = self.metadata_distributor.get_var('uvw_g')
    self.uvw_w = self.metadata_distributor.get_var('uvw_w')
    self.pqr_g = self.metadata_distributor.get_var('pqr_g')
    self.pqr_w = self.metadata_distributor.get_var('pqr_w')
    self.rho = self.metadata_distributor.get_var('rho')
    self.rho_0 = self.metadata_distributor.get_var('rho_0')
    self.nu = self.metadata_distributor.get_var('nu')
    self.t_hb = self.metadata_distributor.get_var('t_hb')

  def get_airspeed(self):
    uvw_a = self.uvw_g - self.uvw_w
    self.metadata_distributor.set({'uvw_a':uvw_a})
    return uvw_a

  def get_air_angular_velocity(self):
    pqr_a = self.pqr_g - self.pqr_w
    self.metadata_distributor.set({'pqr_a': pqr_a})
    return pqr_a

  def get_reference_airspeed(self):
    x_cg, y_cg, z_cg = self.xyz_cg
    u_a, v_a, w_a = self.metadata_distributor.get_var('uvw_a')
    p_a, q_a, r_a = self.metadata_distributor.get_var('pqr_a')
    u_a_ref = u_a + r_a*y_cg + q_a*z_cg
    v_a_ref = v_a - p_a*z_cg + r_a*x_cg
    w_a_ref = w_a - q_a*x_cg + p_a*y_cg
    uvw_a_ref = np.array([u_a_ref, v_a_ref, w_a_ref])
    self.metadata_distributor.set({'uvw_a_ref': uvw_a_ref})
    return uvw_a_ref

  def get_true_airspeed(self):
    v_tas = np.linalg.norm(self.metadata_distributor.get_var('uvw_a_ref'))
    self.metadata_distributor.set({'v_tas': v_tas})
    return v_tas

  def get_attack_angle(self):
    u_a_ref, v_a_ref, w_a_ref = self.metadata_distributor.get_var('uvw_a_ref')    
    alpha = np.arctan(w_a_ref/u_a_ref)
    self.metadata_distributor.set({'alpha': alpha})
    return alpha
  
  def get_slip_angle(self):
    u_a_ref, v_a_ref, w_a_ref = self.metadata_distributor.get_var('uvw_a_ref')
    v_tas = self.metadata_distributor.get_var('v_tas')
    beta = np.arcsin(v_a_ref/v_tas)
    self.metadata_distributor.set({'beta': beta})
    return beta

  def get_mach_number(self):
    v_tas = self.metadata_distributor.get_var('v_tas')
    c_s = self.metadata_distributor.get_var('c_s')
    machn = v_tas/c_s
    self.metadata_distributor.set({'machn': machn})
    return machn
  
  def get_reynolds_number(self):
    # requires atmosphere module, flight-body dimensions
    char_l = self.metadata_distributor.get_var('char_l')
    v_tas = self.metadata_distributor.get_var('v_tas')
    re = self.rho * v_tas * char_l / self.nu
    self.metadata_distributor.set({'re': re})
    return re
  
  def get_dynamic_pressure(self):
    v_tas = self.metadata_distributor.get_var('v_tas')
    q_inf = 0.5*self.rho*v_tas**2
    self.metadata_distributor.set({'q_inf': q_inf})
    return q_inf

  def get_equivalent_airspeed(self):
    v_tas = self.metadata_distributor.get_var('v_tas')
    v_eas = (self.rho/self.rho_0 * v_tas)
    self.metadata_distributor.set({'v_eas': v_eas})
    return v_eas

  def get_ned_airspeed(self):
    uvw_a_ref = self.metadata_distributor.get_var('uvw_a_ref')  
    uvw_a_h = np.matmul(self.t_hb.T, uvw_a_ref)
    self.metadata_distributor.set({'uvw_a_h': uvw_a_h})
    return uvw_a_h
  
  def get_ned_flight_path_angle(self):
    u_a_h, v_a_h, w_a_h = self.metadata_distributor.get_var('uvw_a_h')
    v_tas = self.metadata_distributor.get_var('t_as')
    gamma_a = np.arcsin(-w_a_h/v_tas)
    self.metadata_distributor.set({'gamma_a': gamma_a})
    return gamma_a
  
  def get_ned_azimuth(self):
    u_a_h, v_a_h, w_a_h = self.metadata_distributor.get_var('uvw_a_h')
    zai_a = np.arctan(v_a_h/u_a_h)
    self.metadata_distributor.set({'zai_a': zai_a})
    return zai_a

  # get dimensionless quantities
  def get_dimless_attack_rate(self):
    # TO-DO time derivatives
    flightbody_l = self.metadata_distributor.get_var('flightbody_l')
    alpha_dot = self.metadata_distributor.get_var('alpha_dot')
    v_tas = self.metadata_distributor.get_var('v_tas')
    alpha_dot_hat = alpha_dot*flightbody_l/(2*v_tas)
    self.metadata_distributor.set({'alpha_dot_hat': alpha_dot_hat})
    return alpha_dot_hat
  
  def get_dimless_slip_rate(self):
    # TO-DO time derivatives
    flightbody_w = self.metadata_distributor.get_var('flightbody_w')
    beta_dot = self.metadata_distributor.get_var('beta_dot')
    v_tas = self.metadata_distributor.get_var('v_tas')
    beta_dot_hat = beta_dot*flightbody_w/(2*v_tas)
    self.metadata_distributor.set({'beta_dot_hat': beta_dot_hat})
    return beta_dot_hat

  def get_dimless_air_angular_velocity(self):
    p_a, q_a, r_a = self.metadata_distributor.get_var('pqr_a')
    v_tas = self.metadata_distributor.get_var('v_tas')
    flightbody_w = self.metadata_distributor.get_var('flightbody_w')
    flightbody_l = self.metadata_distributor.get_var('flightbody_l')
    p_a_hat = p_a*flightbody_w/(2*v_tas)
    q_a_hat = q_a*flightbody_l/(2*v_tas)
    r_a_hat = r_a*flightbody_w/(2*v_tas)
    pqr_a_hat = np.array([p_a_hat, q_a_hat, r_a_hat])
    self.metadata_distributor.set({'pqr_a_hat': pqr_a_hat})
    return pqr_a_hat