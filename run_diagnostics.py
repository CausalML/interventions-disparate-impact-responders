import matplotlib as mpl
mpl.use('Agg')
from xauc import *
import matplotlib.pyplot as plt
from fair_policies import *


XTY_locs = [ 'data/crepon/X_pub_c.csv', 'data/crepon/T_pub_c.csv', 'data/crepon/Y_pub_c.csv', 'data/crepon/grf_pred_crepon.csv']
[X,T,Y,tau_hat,f_1pred,f_0pred, fdelta] = read_data( XTY_locs )

tau_hat = cf_pred.values.flatten()
eta_ = 0.02;upper_fdelta = eta_*np.ones(len(A))
n_perc = 100; n_etas = 5
etas = np.linspace(0.001, 0.03, n_etas)
[res_lower_, res_upper_] = get_tpr_disps_over_eta(etas, tau_hat, n_perc, fdelta, A, upper_fdelta, plotting=True, vbs = 5)

[res_lower_tnr, res_upper_tnr] = get_tpr_disps_over_eta(etas, tau_hat, n_perc, fdelta, A, upper_fdelta, plotting=True, vbs = 0, disparity = tnr_disparity)
plt.title('TNR disparity')

[tprs, tnrs, cate_pctiles] = get_tnr_tpr(tau_hat, n_perc, fdelta, A)
classes = ['a', 'b']
plot_ROC_overA(tnrs, tprs, classes, A)
