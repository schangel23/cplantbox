/*
* PiafMunch (v.2) -- Implementing and Solving the Munch model of phloem sap flow in higher plants
*
* Copyright (C) 2004-2019 INRA
*
* Author: A. Lacointe, UMR PIAF, Clermont-Ferrand, France
*
* File: odepack.cpp
*
* This file is part of PiafMunch. PiafMunch is free software: you can redistribute it and/or
* modify it under the terms of the GNU General Public License version 3.0 as published by
* the Free Software Foundation and appearing in the file LICENSE.GPL included in the
* packaging of this file. Please  review the following information to ensure the GNU
* General Public License version 3.0  requirements will be met:
* http://www.gnu.org/copyleft/gpl.html.
*
* PiafMunch is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
* without even the implied warranty of FITNESS FOR A PARTICULAR PURPOSE.
* See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with PiafMunch.
* If not, see <http://www.gnu.org/licenses/>.
*
-----------------------------------------------------------------------------------------------------------------------------------*/

#include <PiafMunch/runPM.h>
#include <time.h>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

// La fonction suivante "habille" un Fortran_vector en N_Vector, i.e. "cree" un N_Vector
// qui PARTAGE LES MEMES DONNEES (atributs v_ et NV_DATA_S, resp.)  que le Fortran_vector d'origine.
// Ceci permet d'appliquer "in-place" a un Fortran_vector les fonctions appliquables a un N_Vector :
//
N_Vector InPlace_NVector(Fortran_vector &v) {
         double* v_ = InPlace_Array(v);
		return(N_VMake_Serial((sunindextype)v_[0], (realtype*)(v_ + 1)));
}

int check_flag(void *flagvalue, string funcname_, int opt);   // utilise en 'verbose' dans cvode
const time_t current = time(NULL) ;
Fortran_vector y ;
void* cvode_mem ;        // espace de travail du solveur; eventuellement utilise par la fonction aux()
void* arkode_mem;        // espace de travail du solveur; eventuellement utilise par la fonction aux()

void (*ff)(double t ,double* y, double* ydot) ;

inline int ffff(realtype t, N_Vector yy, N_Vector yydot, void *f_data) {
// Forme de f compatible avec cvode: noter decalage des indices entre double* y et NV_DATA_S(N_Vector y)
    ff(t, NV_DATA_S(yy) - 1, NV_DATA_S(yydot) - 1) ;
	return 0;
}

// les 2 fonctions suivantes sont a reformuler avec fonctions 'synonymes' a traiter comme f
// pour devenir independantes du nom d'appel ('rootfind'...)
extern void rootfind(double t, double* y, double* g);       // (optionnel) definit d'eventuelles equations g(t,y)=0 a resoudre

inline int gg(realtype t, N_Vector yy, realtype *g, void *g_data) {  // Idem pour rootfind / gg :
    rootfind(t, NV_DATA_S(yy) - 1, g-1) ; // retourne le nb d'equations dans la var. n
    return 0;
}

SUNLinearSolver LS(NULL);
extern int Nt, Nc;
extern vector<int> I_Upflow, I_Downflow;
extern std::weak_ptr<PhloemFlux> phloem_;
extern double T;
extern Fortran_vector vol_ST, vol_ParApo, r_ST, r_ST_ref, Q_Grmax, Q_Rmmax, Q_Exudmax, krm2, len_leaf;
extern Fortran_vector Psi_Xyl;

static inline sunindextype PM_state_index(int group, int node) {
	return (sunindextype)(group * Nt + node - 1);
}

static void PM_add_entry(vector<set<sunindextype> > &rows_by_col, sunindextype row, sunindextype col) {
	if ((col >= 0) && (col < (sunindextype)rows_by_col.size()) && (row >= 0) && (row < (sunindextype)rows_by_col.size())) {
		rows_by_col[col].insert(row);
	}
}

static bool PM_build_sparse_jacobian_pattern(sunindextype neq, vector<set<sunindextype> > &rows_by_col) {
	if ((Nt <= 0) || (neq <= 0) || (neq % Nt != 0)) return false;
	const int ng = (int)(neq / Nt);
	if (ng < 10) return false;

	rows_by_col.assign((size_t)neq, set<sunindextype>());
	for (sunindextype col = 0; col < neq; col++) PM_add_entry(rows_by_col, col, col);

	for (int node = 1; node <= Nt; node++) {
		const sunindextype qst_col = PM_state_index(0, node);
		PM_add_entry(rows_by_col, PM_state_index(0, node), qst_col);
		PM_add_entry(rows_by_col, PM_state_index(1, node), qst_col);
		PM_add_entry(rows_by_col, PM_state_index(2, node), qst_col);
		PM_add_entry(rows_by_col, PM_state_index(3, node), qst_col);
		PM_add_entry(rows_by_col, PM_state_index(4, node), qst_col);
		PM_add_entry(rows_by_col, PM_state_index(5, node), qst_col);
		PM_add_entry(rows_by_col, PM_state_index(8, node), qst_col);

		const sunindextype qmeso_col = PM_state_index(1, node);
		PM_add_entry(rows_by_col, PM_state_index(0, node), qmeso_col);
		PM_add_entry(rows_by_col, PM_state_index(1, node), qmeso_col);
		PM_add_entry(rows_by_col, PM_state_index(7, node), qmeso_col);

		const sunindextype q_s_meso_col = PM_state_index(7, node);
		PM_add_entry(rows_by_col, PM_state_index(1, node), q_s_meso_col);
		PM_add_entry(rows_by_col, PM_state_index(7, node), q_s_meso_col);

		const sunindextype q_s_st_col = PM_state_index(8, node);
		PM_add_entry(rows_by_col, PM_state_index(0, node), q_s_st_col);
		PM_add_entry(rows_by_col, PM_state_index(8, node), q_s_st_col);
		PM_add_entry(rows_by_col, PM_state_index(9, node), q_s_st_col);
	}

	for (int edge = 1; edge <= Nc; edge++) {
		const int up = I_Upflow[edge];
		const int down = I_Downflow[edge];
		const sunindextype up_row = PM_state_index(0, up);
		const sunindextype down_row = PM_state_index(0, down);
		const sunindextype up_col = PM_state_index(0, up);
		const sunindextype down_col = PM_state_index(0, down);
		PM_add_entry(rows_by_col, up_row, up_col);
		PM_add_entry(rows_by_col, down_row, up_col);
		PM_add_entry(rows_by_col, up_row, down_col);
		PM_add_entry(rows_by_col, down_row, down_col);
	}
	return true;
}

static sunindextype PM_sparse_jacobian_nnz(sunindextype neq) {
	vector<set<sunindextype> > rows_by_col;
	if (!PM_build_sparse_jacobian_pattern(neq, rows_by_col)) return neq;
	sunindextype ntnz = 0;
	for (size_t col = 0; col < rows_by_col.size(); col++) ntnz += (sunindextype)rows_by_col[col].size();
	return ntnz;
}

static inline double PM_pos_deriv(double value, double scale) {
	// Right-derivative at the max(0, value) clipping boundary: include value=0
	// in the active branch so the Jacobian sees what Newton sees when it probes
	// y from 0 into y>0. Strict ">" gave a wrong (zero) slope at y=0 while the
	// RHS exposed the Vmax/kM, kHyd, k_mucil contributions on the y>0 side,
	// causing CVODE BDF h-collapse on non-uniform restart states (step-2 hang).
	return (value >= 0.) ? scale : 0.;
}

static inline void PM_add_value(map<pair<sunindextype, sunindextype>, realtype> &values, sunindextype row, sunindextype col, realtype value) {
	values[pair<sunindextype, sunindextype>(row, col)] += value;
}

static double PM_viscosity_resistance_deriv(double c, double r_ref, const shared_ptr<PhloemFlux> &phloem) {
	if (!phloem || !phloem->update_viscosity_ || c <= 0.) return 0.;
	const double tdc = T - 273.15;
	const double d_eau = (999.83952 + tdc * (16.952577 + tdc * (-0.0079905127 + tdc * (-0.000046241757 + tdc * (0.00000010584601 + tdc * (-0.00000000028103006)))))) / (1 + 0.016887236 * tdc);
	const double si_phi = (30 - tdc) / (91 + tdc);
	const double d = c * 342.3 + d_eau;
	const double dd_dc = 342.3;
	const double s0 = (100. * 342.30 * c) / d;
	const double ds0_dc = (100. * 342.30 * (d - c * dd_dc)) / (d * d);
	const double denom = 1900. - 18. * s0;
	const double si = s0 / denom;
	const double dsi_dc = (ds0_dc * denom + 18. * s0 * ds0_dc) / (denom * denom);
	const double exponent = (22.46 * si) - 0.114 + (si_phi * (1.1 + 43.1 * pow(si, 1.25)));
	const double mu = pow(10., exponent) / (24. * 60. * 60.) / 100. / 1000.;
	const double dexponent_dc = (22.46 + si_phi * 43.1 * 1.25 * pow(si, 0.25)) * dsi_dc;
	return r_ref * mu * log(10.) * dexponent_dc;
}

static void PM_build_analytic_jacobian_values(realtype *y_data, map<pair<sunindextype, sunindextype>, realtype> &values) {
	shared_ptr<PhloemFlux> phloem = phloem_.lock();
	if (!phloem) return;
	const double RT = 83.14 * T;
	const double q10fac = pow(phloem->Q10, (T - 273.15 - phloem->TrefQ10) / 10.);

	for (int i = 1; i <= Nt; i++) {
		const int zi = i - 1;
		const double qst = y_data[PM_state_index(0, i)];
		const double qmeso = y_data[PM_state_index(1, i)];
		const double q_s_meso = y_data[PM_state_index(7, i)];
		const double q_s_st = y_data[PM_state_index(8, i)];
		const double vol_st = vol_ST[i];
		const double vol_meso = vol_ParApo[i];
		const double c0 = std::max(0., qst / vol_st);
		const double dc0 = PM_pos_deriv(qst, 1. / vol_st);
		// F.6.B: vol_meso==0 at root nodes (no parenchyma apoplast). Avoid 1/0 -> inf
		// propagation into dfl_dqmeso. RHS PiafMunch2.cpp:206 already gates the
		// mesophyll starch block with `if (vol_ParApo[i]>0)`, so the Jacobian
		// must mirror that exclusion.
		const double cmeso_raw = (vol_meso > 0.) ? (qmeso / vol_meso) : 0.;
		const double cmeso = std::max(0., cmeso_raw);
		const double dcmeso = (vol_meso > 0.) ? PM_pos_deriv(cmeso_raw, 1. / vol_meso) : 0.;

		const double den_st = phloem->kM_S_ST + c0;
		double dstarch_st_dqst = 0.;
		if (den_st != 0. && qst >= 0.) dstarch_st_dqst += phloem->Vmax_S_ST * (den_st - qst * dc0) / (den_st * den_st);
		dstarch_st_dqst += phloem->k_S_ST * vol_st * dc0;
		double dstarch_st_dqsst = (q_s_st >= 0.) ? -phloem->kHyd_S_ST : 0.;
		double mucil = std::max(0., phloem->k_mucil_[zi] * q_s_st);
		double dmucil_dqsst = ((q_s_st >= 0.) && (phloem->k_mucil_[zi] > 0.)) ? phloem->k_mucil_[zi] : 0.;
		double starch_st = 0.;
		if (den_st != 0.) starch_st += phloem->Vmax_S_ST * std::max(0., qst) / den_st;
		starch_st += phloem->k_S_ST * (c0 - phloem->C_targ) * vol_st - phloem->kHyd_S_ST * std::max(0., q_s_st);
		if ((q_s_st <= 0.) && (starch_st < 0.)) {
			dstarch_st_dqst = 0.;
			dstarch_st_dqsst = 0.;
			dmucil_dqsst = 0.;
			mucil = 0.;
		}

		const double den_meso = phloem->kM_S_Mesophyll + cmeso;
		double dstarch_meso_dqmeso = 0.;
		double dstarch_meso_dqsmeso = 0.;
		double starch_meso = 0.;
		if (vol_meso > 0.) {
			if (den_meso != 0. && qmeso >= 0.) dstarch_meso_dqmeso += phloem->Vmax_S_Mesophyll * (den_meso - qmeso * dcmeso) / (den_meso * den_meso);
			dstarch_meso_dqmeso += phloem->k_S_Mesophyll * vol_meso * dcmeso;
			dstarch_meso_dqsmeso = -phloem->kHyd_S_Mesophyll;
			if (den_meso != 0.) starch_meso += phloem->Vmax_S_Mesophyll * std::max(0., qmeso) / den_meso;
			starch_meso += -phloem->kHyd_S_Mesophyll * q_s_meso + phloem->k_S_Mesophyll * (cmeso - phloem->C_targMesophyll) * vol_meso;
			if ((q_s_meso <= 0.) && (starch_meso < 0.)) {
				dstarch_meso_dqmeso = 0.;
				dstarch_meso_dqsmeso = 0.;
			}
		}

		const double load_den = phloem->Mloading + cmeso;
		const double exp_load = exp(-c0 * phloem->beta_loading);
		const double load_a = phloem->Vmaxloading * len_leaf[i];
		const double fl = (load_den != 0.) ? load_a * cmeso / load_den * exp_load : 0.;
		const double dfl_dqmeso = (load_den != 0.) ? load_a * phloem->Mloading / (load_den * load_den) * exp_load * dcmeso : 0.;
		const double dfl_dqst = -phloem->beta_loading * fl * dc0;
		const double cuse = std::max(0., c0 - phloem->CSTimin);
		const double dcuse = (c0 >= phloem->CSTimin) ? dc0 : 0.;
		const double csoil = (zi < (int)phloem->Csoil_node.size()) ? phloem->Csoil_node[zi] : phloem->CsoilDefault;
		const double dc_delta = (cuse >= csoil) ? dcuse : 0.;
		const double drmmax = krm2[i] * q10fac * dcuse;
		const double rmmax = (Q_Rmmax[i] + krm2[i] * cuse) * q10fac;
		const double dexud = Q_Exudmax[i] * dc_delta;
		const double fu_den = cuse + phloem->KMfu;
		const double fu_base = rmmax + Q_Grmax[i];
		const double fu = (fu_den != 0.) ? fu_base * cuse / fu_den : 0.;
		const double dfu = (fu_den != 0.) ? (drmmax * cuse / fu_den + fu_base * phloem->KMfu * dcuse / (fu_den * fu_den)) : 0.;
		const double drm = (fu <= rmmax) ? dfu : drmmax;
		const double growth_raw = fu - std::min(fu, rmmax);
		const double dgrowth = ((growth_raw >= 0.) && (growth_raw <= Q_Grmax[i])) ? (dfu - drm) : 0.;

		PM_add_value(values, PM_state_index(0, i), PM_state_index(0, i), dfl_dqst - dfu - dexud - dstarch_st_dqst);
		PM_add_value(values, PM_state_index(0, i), PM_state_index(1, i), dfl_dqmeso);
		PM_add_value(values, PM_state_index(0, i), PM_state_index(8, i), -dstarch_st_dqsst);
		PM_add_value(values, PM_state_index(1, i), PM_state_index(0, i), -dfl_dqst);
		PM_add_value(values, PM_state_index(1, i), PM_state_index(1, i), -dfl_dqmeso - dstarch_meso_dqmeso);
		PM_add_value(values, PM_state_index(1, i), PM_state_index(7, i), -dstarch_meso_dqsmeso);
		PM_add_value(values, PM_state_index(2, i), PM_state_index(0, i), drm);
		PM_add_value(values, PM_state_index(3, i), PM_state_index(0, i), dexud);
		PM_add_value(values, PM_state_index(4, i), PM_state_index(0, i), dgrowth);
		PM_add_value(values, PM_state_index(5, i), PM_state_index(0, i), drmmax);
		PM_add_value(values, PM_state_index(7, i), PM_state_index(1, i), dstarch_meso_dqmeso);
		PM_add_value(values, PM_state_index(7, i), PM_state_index(7, i), dstarch_meso_dqsmeso);
		PM_add_value(values, PM_state_index(8, i), PM_state_index(0, i), dstarch_st_dqst);
		PM_add_value(values, PM_state_index(8, i), PM_state_index(8, i), dstarch_st_dqsst - dmucil_dqsst);
		PM_add_value(values, PM_state_index(9, i), PM_state_index(8, i), dmucil_dqsst);
	}

	for (int edge = 1; edge <= Nc; edge++) {
		const int up = I_Upflow[edge];
		const int down = I_Downflow[edge];
		const double cup = std::max(0., y_data[PM_state_index(0, up)] / vol_ST[up]);
		const double cdown = std::max(0., y_data[PM_state_index(0, down)] / vol_ST[down]);
		const double dcup = PM_pos_deriv(y_data[PM_state_index(0, up)], 1. / vol_ST[up]);
		const double dcdown = PM_pos_deriv(y_data[PM_state_index(0, down)], 1. / vol_ST[down]);
		const double pup = RT * cup + (phloem->usePsiXyl ? Psi_Xyl[up] : 0.);
		const double pdown = RT * cdown + (phloem->usePsiXyl ? Psi_Xyl[down] : 0.);
		const double w = pup - pdown;
		const bool from_up = (w > 0.);
		const double ca = from_up ? cup : cdown;
		const double r = r_ST[edge];
		// Viscosity residue: r_ST = mu(C_amont) * r_ST_ref is frozen in the Jacobian.
		// Including dmu/dC made the maize KLU path leave the empirical 1 h baseline;
		// with C_ST < 0.3 mmol cm-3 here this bounded lagged coefficient keeps Newton stable.
		const double dr_dca = 0.; // PM_viscosity_resistance_deriv(ca, r_ST_ref[edge], phloem)
		const double common = (r != 0.) ? ca / r : 0.;
		const double adv = (r != 0.) ? w / r : 0.;
		const double visc = (r != 0.) ? -w * ca / (r * r) * dr_dca : 0.;
		const double djs_dqup = common * RT * dcup + (from_up ? (adv + visc) * dcup : 0.);
		const double djs_dqdown = -common * RT * dcdown + (!from_up ? (adv + visc) * dcdown : 0.);
		PM_add_value(values, PM_state_index(0, up), PM_state_index(0, up), -djs_dqup);
		PM_add_value(values, PM_state_index(0, down), PM_state_index(0, up), djs_dqup);
		PM_add_value(values, PM_state_index(0, up), PM_state_index(0, down), -djs_dqdown);
		PM_add_value(values, PM_state_index(0, down), PM_state_index(0, down), djs_dqdown);
	}
}

// FD-vs-analytic Jacobian audit. Fires once when PM_AUDIT_AFTER=N is set and the
// jac_call_count reaches N. Compares analytic columns from PM_build_analytic_jacobian_values
// against finite-difference RHS columns on a deterministic 12-column sample (3 nodes
// from each of 4 active groups: Q_ST, Q_M, Q_S_M, Q_S_ST). Reports per-column
// max-rel-err inside the structural pattern and max-abs FD outside the pattern.
static void PM_jacobian_audit(realtype t, realtype *y_data, sunindextype neq,
                               const map<pair<sunindextype, sunindextype>, realtype> &values,
                               const vector<set<sunindextype> > &rows_by_col) {
	if (Nt < 3) return;
	int nodes[3] = {1, std::max(1, Nt / 2), Nt};
	int groups[4] = {0, 1, 7, 8};
	const char *group_names[4] = {"Q_ST", "Q_M", "Q_S_M", "Q_S_ST"};

	vector<double> y_local((size_t)neq);
	vector<double> f0((size_t)neq, 0.), f1((size_t)neq, 0.);

	if (!ff) { cout << "[JAC-AUDIT] ff is NULL — skipping" << endl; return; }
	for (sunindextype k = 0; k < neq; k++) y_local[(size_t)k] = y_data[k];
	ff(t, &y_local[0] - 1, &f0[0] - 1);

	double max_rel_err_grp[4] = {0., 0., 0., 0.};
	double max_abs_oop_grp[4] = {0., 0., 0., 0.};
	int    oop_count_grp[4]   = {0, 0, 0, 0};

	cout << "[JAC-AUDIT] t=" << t << " neq=" << neq << " Nt=" << Nt << endl;

	for (int gi = 0; gi < 4; gi++) {
		const int g = groups[gi];
		for (int ni = 0; ni < 3; ni++) {
			const int node = nodes[ni];
			const sunindextype col = PM_state_index(g, node);
			if (col < 0 || col >= neq) continue;
			for (sunindextype k = 0; k < neq; k++) y_local[(size_t)k] = y_data[k];
			const double y_orig = y_local[(size_t)col];
			const double eps = 1e-7 * std::max(1e-3, std::abs(y_orig));
			y_local[(size_t)col] = y_orig + eps;
			ff(t, &y_local[0] - 1, &f1[0] - 1);

			const int in_pattern_n = (col < (sunindextype)rows_by_col.size()) ? (int)rows_by_col[(size_t)col].size() : 0;
			double max_abs_err_col = 0., max_rel_err_col = 0.;
			sunindextype worst_row = -1;
			int oop_n = 0;
			double max_abs_oop = 0.;
			sunindextype worst_oop_row = -1;
			double worst_pair_an = 0., worst_pair_fd = 0.;

			for (sunindextype row = 0; row < neq; row++) {
				const double fd = (f1[(size_t)row] - f0[(size_t)row]) / eps;
				const bool in_pattern = (col < (sunindextype)rows_by_col.size()) && (rows_by_col[(size_t)col].count(row) > 0);
				if (in_pattern) {
					map<pair<sunindextype, sunindextype>, realtype>::const_iterator it = values.find(pair<sunindextype, sunindextype>(row, col));
					const double analytic = (it == values.end()) ? 0. : (double)it->second;
					const double abs_err = std::abs(analytic - fd);
					const double scale = std::max(std::abs(analytic), std::abs(fd));
					const double rel_err = (scale > 1e-30) ? (abs_err / scale) : 0.;
					if (abs_err > max_abs_err_col) {
						max_abs_err_col = abs_err;
						worst_row = row;
						worst_pair_an = analytic;
						worst_pair_fd = fd;
					}
					if (rel_err > max_rel_err_col) max_rel_err_col = rel_err;
				} else {
					if (std::abs(fd) > 1e-12) {
						oop_n++;
						if (std::abs(fd) > max_abs_oop) {
							max_abs_oop = std::abs(fd);
							worst_oop_row = row;
						}
					}
				}
			}
			if (max_rel_err_col > max_rel_err_grp[gi]) max_rel_err_grp[gi] = max_rel_err_col;
			if (max_abs_oop > max_abs_oop_grp[gi]) max_abs_oop_grp[gi] = max_abs_oop;
			oop_count_grp[gi] += oop_n;
			cout << "[JAC-AUDIT col=" << col << " grp=" << group_names[gi] << " node=" << node
			     << " y=" << y_orig << " eps=" << eps
			     << "] pattern_n=" << in_pattern_n
			     << " max_abs_err=" << max_abs_err_col
			     << " max_rel_err=" << max_rel_err_col
			     << " worst_row=" << worst_row
			     << " an=" << worst_pair_an << " fd=" << worst_pair_fd
			     << " oop_n=" << oop_n
			     << " max_abs_oop=" << max_abs_oop
			     << " oop_row=" << worst_oop_row << endl;
		}
	}

	cout << "[JAC-AUDIT SUMMARY] grouped_max_rel_err"
	     << " Q_ST=" << max_rel_err_grp[0]
	     << " Q_M=" << max_rel_err_grp[1]
	     << " Q_S_M=" << max_rel_err_grp[2]
	     << " Q_S_ST=" << max_rel_err_grp[3] << endl;
	cout << "[JAC-AUDIT SUMMARY] grouped_max_abs_oop"
	     << " Q_ST=" << max_abs_oop_grp[0]
	     << " Q_M=" << max_abs_oop_grp[1]
	     << " Q_S_M=" << max_abs_oop_grp[2]
	     << " Q_S_ST=" << max_abs_oop_grp[3] << endl;
	cout << "[JAC-AUDIT SUMMARY] grouped_oop_count"
	     << " Q_ST=" << oop_count_grp[0]
	     << " Q_M=" << oop_count_grp[1]
	     << " Q_S_M=" << oop_count_grp[2]
	     << " Q_S_ST=" << oop_count_grp[3] << endl;
	cout.flush();
}

/* Other Constants pour calcul KLU_DQ_Jac : */
#define MIN_INC_MULT RCONST(1000.0)
#define ZERO         RCONST(0.0)
#define ONE          RCONST(1.0)
#define TWO          RCONST(2.0)

int Jac_(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
	sunindextype *colptrs = SUNSparseMatrix_IndexPointers(J);
	sunindextype *rowvals = SUNSparseMatrix_IndexValues(J);
	realtype *data = SUNSparseMatrix_Data(J);
	realtype *y_data = N_VGetArrayPointer(y);
	sunindextype j, N, ntnz = 0;
	vector<set<sunindextype> > rows_by_col;
	map<pair<sunindextype, sunindextype>, realtype> values;

	SUNMatZero(J);
	N = SM_COLUMNS_S(J);
	assert(SM_ROWS_S(J) == N);
	assert(NV_LENGTH_S(y) == N);
	assert(NV_LENGTH_S(fy) == N);
	if (!PM_build_sparse_jacobian_pattern(N, rows_by_col)) return -1;

	const sunindextype structural_nnz = PM_sparse_jacobian_nnz(N);
	if (SM_NNZ_S(J) < structural_nnz) {
		SUNSparseMatrix_Reallocate(J, structural_nnz);
		colptrs = SUNSparseMatrix_IndexPointers(J);
		rowvals = SUNSparseMatrix_IndexValues(J);
		data = SUNSparseMatrix_Data(J);
	}

	PM_build_analytic_jacobian_values(y_data, values);

	{
		static int jac_call_count = 0;
		static bool audit_done = false;
		jac_call_count++;
		const char *log_every_s = std::getenv("PM_JAC_LOG_EVERY");
		if (log_every_s) {
			const int log_every = std::atoi(log_every_s);
			if (log_every > 0 && (jac_call_count % log_every) == 0) {
				cout << "[JAC-COUNT] n=" << jac_call_count << " t=" << t << endl;
				cout.flush();
			}
		}
		const char *audit_after_s = std::getenv("PM_AUDIT_AFTER");
		if (audit_after_s && !audit_done) {
			const int audit_after = std::atoi(audit_after_s);
			if (audit_after > 0 && jac_call_count >= audit_after) {
				PM_jacobian_audit(t, y_data, N, values, rows_by_col);
				audit_done = true;
			}
		}
	}

	for (j = 0; j < N; j++) {
		colptrs[j] = ntnz;
		for (set<sunindextype>::const_iterator row = rows_by_col[j].begin(); row != rows_by_col[j].end(); ++row) {
			map<pair<sunindextype, sunindextype>, realtype>::const_iterator value = values.find(pair<sunindextype, sunindextype>(*row, j));
			rowvals[ntnz] = *row;
			data[ntnz] = (value == values.end()) ? ZERO : value->second;
			ntnz++;
		}
	}
	colptrs[N] = ntnz;
	if (LS) {
		SUNLinSol_KLUReInit(LS, J, ntnz, SUNKLU_REINIT_PARTIAL);
	}
	return(0);
}


int cvode_direct(void(*f)(double,double*,double*), Fortran_vector& y, Fortran_vector &T, void(*aux)(double,double*), Fortran_vector& atol, Fortran_vector& rtol, int solver,
	     int nbVar_dot, Fortran_vector** Var_primitive, Fortran_vector** Var_dot, bool verbose, bool STALD, void(*rootfind)(double, double*, double*), int nrootfns, int mu, int ml) {
	ff = f ;
  int itol = 1 ;
/* 	itol = 1 (= CV_SS) : atol and rtol tous 2 scalaires ;
		itol = 2 (= CV_SV) : atol vecteur, rtol scalaire ;
		itol = 3 (= CV_WF) : ewt[i] defini par une fonction utilisateur (non encore implemente ; cf. doc CVODE p.29) */
  if (atol.size() > 1) itol ++ ;
  if (rtol.size() > 1) itol += 2 ;
  if (itol == 3) {_LogMessage("Erreur itol = CV_WF : cas non encore traite par cvode") ; return -1;}
  if (itol == 4) {_LogMessage("Erreur itol = 4 (atol et rtol tous deux vecteurs): non traite par cvode") ; return -1 ;}
  int neq = y.size();							 // taille du probleme = nb d'equations = nb de variables y[i]
  int nbt = T.size() ;							 // nombre de temps ou l'on veut la solution, y compris t0 = T[1]
  int i, j, flag, flagr ;
  double t, tout, t_sav, dt ;
  N_Vector abstol(InPlace_NVector(atol)) ;
  N_Vector yy(InPlace_NVector(y));              // "habille" le Fortran_vector y en N_Vector sans occuper de memoire suppl.
  SUNMatrix A;
  double* y_ = InPlace_Array(y);	// y_ = y.v_
	t = t_sav = T[1] ;
   Fortran_vector** Var_primitive_sav ;
   if (aux != NULL) aux(T[1], y_) ; // instant t0 = T[1]
  Update_Output(true) ;
  if (nbVar_dot) {
	  assert(Var_primitive) ;
	  assert(Var_dot) ;
	  for (j = 0 ; j < nbVar_dot ; j ++) {
		  assert(Var_primitive[j]) ;
		  assert(Var_dot[j]) ;
	  }
	  assert(Var_primitive[nbVar_dot] == NULL) ; assert(Var_dot[nbVar_dot] == NULL) ; // previendra l'utilisateur si on a declare moins de derivees que l'on n'en a dimensionnees
	  Var_primitive_sav = new Fortran_vector*[nbVar_dot] ;
	  for (j = 0 ; j < nbVar_dot ; j ++) 	  Var_primitive_sav[j] = new Fortran_vector(Var_primitive[j]->size()) ;
  }
  cvode_mem = CVodeCreate(CV_BDF);    // init. le solveur avec la methode d'integr. BDF (et la resol. de type Newton)
    if (check_flag(cvode_mem, "CVodeCreate", 0)) {_LogMessage("erreur CVodeCreate") ; return -1 ; }
  double *y_data_ptr = NV_DATA_S(yy);
  for (sunindextype k = 0; k < neq; k++) if (y_data_ptr[k] < 0.) y_data_ptr[k] = 0.;
  flag = CVodeInit(cvode_mem, ffff, T[1], yy); // alloue l'espace de travail du solveur
    if (check_flag(&flag, "CVodeInit", 1)) {_LogMessage("erreur CVodeInit") ; return -1 ; }
  N_Vector constraints = N_VNew_Serial(neq);
  realtype *c = NV_DATA_S(constraints);
  for (sunindextype k = 0; k < neq; k++) c[k] = 1.0;
  flag = CVodeSetConstraints(cvode_mem, constraints);
  N_VDestroy(constraints);
    if (check_flag(&flag, "CVodeSetConstraints", 1)) return(1);

/* Call CVodeSVtolerances to specify the scalar relative tolerance
* and vector absolute tolerances */
	flag = CVodeSVtolerances(cvode_mem, rtol[1], abstol);
	if (check_flag(&flag, "CVodeSVtolerances", 1)) return(1);

  if (STALD) {                // algorithme de detection/correction de stabilite aux ordres > 2 (integration par methode BDF)
      flag = CVodeSetStabLimDet(cvode_mem, SUNTRUE);    // FALSE par defaut
        if (check_flag(&flag, "CVodeSetStabLimDet", 1)) {_LogMessage("erreur CVodeSetStabLimDet") ; return -1 ; }
  }
  int* rootsfound = NULL ;
  if (rootfind != NULL) {
      if(nrootfns < 1) {_LogMessage("erreur RootFind : nrootfns =< 0 eq. a resoudre !") ; return -1 ; }
      rootsfound = new int[nrootfns] ;
	  flag = CVodeRootInit(cvode_mem, nrootfns, gg);
	  if (check_flag(&flag, "CVodeRootInit", 1)) {_LogMessage("erreur CVodeRootInit") ; return -1 ; }
  }
  if (solver == DIAG) {
	  flag = CVDiag(cvode_mem);
	  if (check_flag(&flag, "CVDiag", 1)) { _LogMessage("erreur CVDiag"); return -1; }
  } else {
	  if (solver == DENSE) {
		  /* Create dense SUNMatrix for use in linear solves */
		  A = SUNDenseMatrix(neq, neq);
		  if (!(SM_ROWS_S(A) == neq)) { _LogMessage("erreur SunDenseMatrix"); return -1; }
		  /* Create dense SUNLinearSolver object for use by CVode */
		  LS = SUNLinSol_Dense(yy, A);
		  if (!LS) { _LogMessage("erreur SUNLinSol_Dense"); return -1; }
	  } else {
		  if (solver == BAND) {
			  A = SUNBandMatrix(neq, mu, ml);
			  if (!(SM_ROWS_S(A) == neq)) { _LogMessage("erreur SunBandMatrix"); return -1; }
			  /* Create banded SUNLinearSolver object for use by CVode */
			  LS = SUNLinSol_Band(yy, A);
			  if (!LS) { _LogMessage("erreur SUNLinSol_Band"); return -1; }
		  } else {
					if (solver == KLU) {// cree les 3 espaces memoire (KLU::Ax, Ai, Ap) dimensionnes ; on pourra les redimensionner ulterieurement...
						A = SUNSparseMatrix(neq, neq, PM_sparse_jacobian_nnz(neq), CSC_MAT);// 3eme arg : structural nnz for the PiafMunch sparse Jacobian
					if (check_flag((void *)A, "SUNSparseMatrix", 0)) return(1);
					/* Create KLU solver object for use by CVode */
					LS = SUNLinSol_KLU(yy, A);
					if (check_flag((void *)LS, "SUNLinSol_KLU", 0)) return(1);
				}
				else { (void)sprintf(message, "%d : nom de solveur inconnu", solver); _LogMessage(message); return -1; }
		  }
	  }
	  /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
	  flag = CVodeSetLinearSolver(cvode_mem, LS, A);
	  if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return(1);
	  if (solver == KLU) {
		  flag = CVodeSetJacFn(cvode_mem, Jac_);
		  if (check_flag(&flag, "CVodeSetJacFn", 1)) return(1); // Non-zero = erreur
	  }
	  }
	  flag = CVodeSetMaxConvFails(cvode_mem, 100) ; // pour prevenir l'erreur de non-convergence (error code = -4), sauf si la convergence n'est effectivement jamais atteinte !
	  if (check_flag(&flag, "CVodeSetMaxConvFails", 1)) return(1);
	  flag = CVodeSetMaxNumSteps(cvode_mem, 10000);
	  if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return(1);
	  for (i = 2 ; i <= nbt ; i++) {						// pour chaque instant ou l'on souhaite la solution
	tout=T[i];
	if (verbose) {
			strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  starting step n#" << i-1 << " (tf = " << tout << ")" << endl ;
			Update_Output(i == 2) ;
	}
	while (t < tout) {
		flag = -9999 ; // bidon
		while (flag != CV_SUCCESS) {
			  flag = CVode(cvode_mem, tout, yy, &t, CV_ONE_STEP); // appel au solveur en mode 'CV_ONE_STEP' // attention : cette instruction MAJ  t, qui peut donc maintenant etre > tout !!
			  if (flag == CV_ROOT_RETURN) {
				 flagr = CVodeGetRootInfo(cvode_mem, rootsfound);
				   if (check_flag(&flagr, "CVodeGetRootInfo", 1)) { _LogMessage(" Erreur CVodeGetRootInfo") ;
						//strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  exiting solver" << endl ; 
						Update_Output() ; return i ;
				   }
				 for (j = 0 ; j < nrootfns ; j++) {
					 if (rootsfound[j] == 1) {
						(void)sprintf(message, "Eq.[%d] : Root Found at t = %g :", j+1, t) ; _LogMessage(message) ;
						Update_Output() ;
					 }
				 }
				 aux(t, y_) ;
			  }
			  else {
				   if (flag != CV_SUCCESS) {
				  //    if (check_flag(&flag, "CVode", 1)) break ; // routine a  reprendre...
					   (void)sprintf(message, "error-flag CVode = %d", flag) ; _LogMessage(message) ;
						//strftime(message, 50, "%H:-%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  exiting solver" << endl ; 
						Update_Output(true) ;
					   return i ;
				  }
			  }
		}
		if (nbVar_dot) {
			dt = t - t_sav;
			for (j = 0; j < nbVar_dot; j++) {
				Var_dot[j]->set((*(Var_primitive[j]) - *(Var_primitive_sav[j])) / dt);
				Var_primitive_sav[j]->set(*(Var_primitive[j]));
			}
			if (t < tout)  t_sav = t;
		}
	} // fin boucle  while(t < tout) : les 8 lignes suivantes sont donc executees si (t > tout) :
		if (aux != NULL) {		// calculs auxiliaires, par ex. sauvegarder var. intermediaires
								// a ce stade t_curr est juste superieur a tout : on interpole y a y_out = y(tout) :
			flag = CVodeGetDky(cvode_mem, tout, 0, yy);
			check_flag(&flag, "CVodeGetDky", 1);
			aux(T[i], y_);
		}
		t_sav = t;
	} // fin boucle i (T[i])
  if (verbose) {  // Print some final statistics :
      long int nst, nsetups, nje, nfeLS, nni, ncfn, netf, nge;
      flag = CVodeGetNumSteps(cvode_mem, &nst);
            check_flag(&flag, "CVodeGetNumSteps", 1);
      flag = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
            check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
      flag = CVodeGetNumErrTestFails(cvode_mem, &netf);
            check_flag(&flag, "CVodeGetNumErrTestFails", 1);
      flag = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
           check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
      flag = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
           check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);
      flag = CVodeGetNumJacEvals(cvode_mem, &nje);
           check_flag(&flag, "CVodeGetNumJacEvals", 1);
      flag = CVodeGetNumRhsEvals(cvode_mem, &nfeLS);
           check_flag(&flag, "CVodeGetNumRhsEvals", 1);
      flag = CVodeGetNumGEvals(cvode_mem, &nge);
           check_flag(&flag, "CVodeGetNumGEvals", 1);
      _LogMessage("\nFinal Statistics:");
      (void)sprintf(message, "nst = %-6ld nsetups = %-6ld nfeLS = %-6ld nje = %ld\n", nst, nsetups, nfeLS, nje);
           _LogMessage(message) ;
      (void)sprintf(message, "nni = %-6ld ncfn = %-6ld netf = %-6ld nge = %ld\n \n", nni, ncfn, netf, nge); _LogMessage(message) ;
  }
  /* Free integrator memory : */
  CVodeFree(&cvode_mem);
  N_VDestroy_Serial(yy) ; N_VDestroy_Serial(abstol) ; // heureusement, ne desallouent pas leurs 'double* NV_DATA_S()' car issus de 'N_VMake_Serial'
  if (rootfind != NULL) delete [] rootsfound ;
  if (nbVar_dot) {
	  for (j = 0 ; j < nbVar_dot ; j ++) 	 delete Var_primitive_sav[j] ;
	delete[] Var_primitive_sav ;
  }
//  _strtime_s(message, 100); cout << "at " << message << " :  exiting solver" << endl ; Update_Output() ;
   //strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  exiting solver" << endl ; 
   Update_Output() ;
	return 0 ;
}


int cvode_spils(void(*f)(double, double*, double*), Fortran_vector &y, Fortran_vector &T, void(*aux)(double, double*), Fortran_vector& atol, Fortran_vector& rtol,
	int solver, int GSType, int prectype, int nbVar_dot, Fortran_vector** Var_primitive, Fortran_vector** Var_dot,
	bool verbose, bool STALD, void(*rootfind)(double, double*, double*), int nrootfns, int mu, int ml, int maxl) {
	ff = f;
	int itol = 1;
	// 		itol = 1 (= CV_SS) : atol and rtol tous 2 scalaires ;
	//		itol = 2 (= CV_SV) : atol vecteur, rtol scalaire ;
	//		itol = 3 (= CV_WF) : ewt[i] defini par une fonction utilisateur (non encore implemente ; cf. doc CVODE p.29)
	if (atol.size() > 1) itol++;
	if (rtol.size() > 1) itol += 2;
	if (itol == 3) { _LogMessage("Erreur itol = CV_WF : cas non encore traite par cvode"); return -1; }
	if (itol == 4) { _LogMessage("Erreur itol = 4 (atol et rtol tous deux vecteurs): non traite par cvode"); return -1; }
	int neq = y.size();							 // taille du probleme = nb d'equations = nb de variables y[i]
	int nbt = T.size();							 // nombre de temps ou l'on veut la solution, y compris t0 = T[1]
	int i, j, flag, flagr;
	double t, tout, t_sav, dt;
	N_Vector yy(InPlace_NVector(y));              // "habille" le Fortran_vector y en N_Vector sans occuper de memoire suppl.
	N_Vector abstol(InPlace_NVector(atol));
	double* y_ = NV_DATA_S(yy) - 1;
	t = t_sav = T[1];
	Fortran_vector** Var_primitive_sav;
	if (aux != NULL) aux(T[1], y_); // instant t0 = T[1]
	Update_Output(true);
	if (nbVar_dot) {
		assert(Var_primitive);
		assert(Var_dot);
		for (j = 0; j < nbVar_dot; j++) {
			assert(Var_primitive[j]);
			assert(Var_dot[j]);
		}
		assert(Var_primitive[nbVar_dot] == NULL); assert(Var_dot[nbVar_dot] == NULL); // previendra l'utilisateur si on a declare moins de derivees que l'on n'en a dimensionnees
		Var_primitive_sav = new Fortran_vector*[nbVar_dot];
		for (j = 0; j < nbVar_dot; j++) 	  Var_primitive_sav[j] = new Fortran_vector(Var_primitive[j]->size());
	}
	cvode_mem = CVodeCreate(CV_BDF);    // init. le solveur avec la methode d'integr. BDF et la resol. de type Newton
	if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) { _LogMessage("erreur CVodeCreate"); return -1; }

	flag = CVodeInit(cvode_mem, ffff, T[1], yy); // alloue l'espace de travail du solveur
	if (check_flag(&flag, "CVodeInit", 1)) { _LogMessage("erreur CVodeInit"); return -1; }

	/* Call CVodeSVtolerances to specify the scalar relative tolerance
	* and vector absolute tolerances */
	flag = CVodeSVtolerances(cvode_mem, rtol[1], abstol);
	if (check_flag(&flag, "CVodeSVtolerances", 1))
	{
		std::cout<<"check_flag(&flag, CVodeSVtolerances, 1) "<< flag <<std::endl;
		return(1);
	}

	if (STALD) {                // algorithme de detection/correction de stabilite aux ordres > 2 (integration par methode BDF)
		flag = CVodeSetStabLimDet(cvode_mem, SUNTRUE);    // FALSE par defaut
		if (check_flag(&flag, "CVodeSetStabLimDet", 1)) { _LogMessage("erreur CVodeSetStabLimDet"); return -1; }
	}
	int* rootsfound = NULL;
	if (rootfind != NULL) {
		if (nrootfns < 1) { _LogMessage("erreur RootFind : nrootfns =< 0 eq. a resoudre !"); return -1; }
		rootsfound = new int[nrootfns];
		flag = CVodeRootInit(cvode_mem, nrootfns, gg);
		if (check_flag(&flag, "CVodeRootInit", 1)) { _LogMessage("erreur CVodeRootInit"); return -1; }
	}
	if (solver == SPGMR) {
		LS = SUNLinSol_SPGMR(yy, prectype, maxl);
		if (!LS) { _LogMessage("erreur SUNLinSol_spgmr"); return -1; }
		flag = SUNLinSol_SPGMRSetGSType(LS, GSType); // Gram-Schmidt orthogonalisation method
		if (check_flag(&flag, "SUNLinSol_SPGMRSetGSType", 1)) { _LogMessage("erreur SUNLinSol_SPGMRSetGSType"); return -1; }
	}
	else {
		if (solver == SPFGMR) {
			LS = SUNLinSol_SPFGMR(yy, prectype, maxl);
			if (!LS) { _LogMessage("erreur SUNLinSol_spfgmr"); return -1; }
			flag = SUNLinSol_SPFGMRSetGSType(LS, GSType); // Gram-Schmidt orthogonalisation method
			if (check_flag(&flag, "SUNLinSol_SPFGMRSetGSType", 1)) { _LogMessage("erreur SUNLinSol_SPFGMRSetGSType"); return -1; }
		}
		else {
			if (solver == SPBCGS) {
				LS = SUNLinSol_SPBCGS(yy, prectype, maxl);
				if (!LS) { _LogMessage("erreur SUNLinSol_spbcgs"); return -1; }
			}
			else {
				if (solver == SPTFQMR) {
					LS = SUNLinSol_SPTFQMR(yy, prectype, maxl);
					if (!LS) { _LogMessage("erreur SUNLinSol_sptfqmr"); return -1; }
				}
				else {
					if (solver == PCG) {
						LS = SUNLinSol_PCG(yy, prectype, maxl);
						if (!LS) { _LogMessage("erreur SUNLinSol_PCG"); return -1; }
					}
					else { (void)sprintf(message, "%d : nom de solveur inconnu", solver); _LogMessage(message); return -1; }
				}
			}
		}
	}
	flag = CVodeSetLinearSolver(cvode_mem, LS, NULL);
	if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return -1;
	if (prectype != PREC_NONE) {
		flag = CVBandPrecInit(cvode_mem, neq, mu, ml);
		cout << "CVBandPrecInit flag = " << flag << endl;
		if (check_flag(&flag, "CVBandPrecInit", 1)) { _LogMessage("erreur CVBandPrecInit"); return -1; }
	}
	  flag = CVodeSetMaxConvFails(cvode_mem, 100) ; // pour prevenir l'erreur de non-convergence (error code = -4), sauf si la convergence n'est effectivement jamais atteinte !
	  if (check_flag(&flag, "CVodeSetMaxConvFails", 1)) return(1);
	  flag = CVodeSetMaxNumSteps(cvode_mem, 10000);
	  if (check_flag(&flag, "CVodeSetMaxNumSteps", 1)) return(1);
	  for (i = 2 ; i <= nbt ; i++) {						// pour chaque instant ou l'on souhaite la solution
	tout=T[i];
	if (verbose) {
			strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  starting step n#" << i-1 << " (tf = " << tout << ")" << endl ;
			Update_Output(i == 2) ;
	}
	while (t < tout) {
		flag = -9999 ; // bidon
		while (flag != CV_SUCCESS) {
			  flag = CVode(cvode_mem, tout, yy, &t, CV_ONE_STEP); // appel au solveur
			  if (flag == CV_ROOT_RETURN) {
				 flagr = CVodeGetRootInfo(cvode_mem, rootsfound);
				   if (check_flag(&flagr, "CVodeGetRootInfo", 1)) { _LogMessage(" Erreur CVodeGetRootInfo") ;
		       			//strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  exiting solver" << endl ; 
						Update_Output() ; return i ;
				   }
				 for (j = 0 ; j < nrootfns ; j++) {
					 if (rootsfound[j] == 1) {
						(void)sprintf(message, "Eq.[%d] : Root Found at t = %g :", j+1, t) ; _LogMessage(message) ;
						Update_Output() ;
					 }
				 }
				 
				std::cout<<"aux for t < tout"<<std::endl;
				aux(t, y_) ;
			  }
			  else {
				  if (flag != CV_SUCCESS) {
				  //    if (check_flag(&flag, "CVode", 1)) break ; // routine a  reprendre...
					   (void)sprintf(message, "error-flag CVode = %d", flag) ; _LogMessage(message) ;
					   	//strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  exiting solver" << endl ; 
						Update_Output() ;
					   return i ;
				  }
			  }
		}
		if (nbVar_dot) {
			dt = t - t_sav;
			for (j = 0; j < nbVar_dot; j++) {
				Var_dot[j]->set((*(Var_primitive[j]) - *(Var_primitive_sav[j])) / dt);
				Var_primitive_sav[j]->set(*(Var_primitive[j]));
			}
			if (t < tout)  t_sav = t;
		}
	  } // fin boucle  while(t < tout) : les 8 lignes suivantes sont donc executees si (t > tout) :
	  if (aux != NULL) {		// calculs auxiliaires, par ex. sauvegarder var. intermediaires
		//if (true){//aux != NULL) {		// calculs auxiliaires, par ex. sauvegarder var. intermediaires
								// a ce stade t_curr est juste superieur a tout : on interpole y a y_out = y(tout) :
		  flag = CVodeGetDky(cvode_mem, tout, 0, yy);
		  check_flag(&flag, "CVodeGetDky", 1);
		  aux(T[i], y_);
	  }
	  t_sav = t;
	} // fin boucle i (T[i])
	if (verbose) {  // Print some final statistics :
      long int nst, nfe, nsetups, nni, ncfn, netf, nge;
      flag = CVodeGetNumSteps(cvode_mem, &nst);
            check_flag(&flag, "CVodeGetNumSteps", 1);
      flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);
            check_flag(&flag, "CVodeGetNumRhsEvals", 1);
      flag = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
            check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
      flag = CVodeGetNumErrTestFails(cvode_mem, &netf);
            check_flag(&flag, "CVodeGetNumErrTestFails", 1);
      flag = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
           check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
      flag = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
           check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);
      flag = CVodeGetNumGEvals(cvode_mem, &nge);
           check_flag(&flag, "CVodeGetNumGEvals", 1);
      _LogMessage("\nFinal Statistics:");
      (void)sprintf(message, "nst (num steps) = %-6ld nfe  (num call to f)= %-6ld nsetups (call to lin solver setup func)= %-6ld\n", nst, nfe, nsetups);
           _LogMessage(message) ;
      (void)sprintf(message, "nni (iter of nonlinear solver) = %-6ld ncfn (non linsolver conv fail)= %-6ld netf (num err test fail) = %-6ld nge (call to root function) = %ld\n \n", nni, ncfn, netf, nge); _LogMessage(message) ;
  }
  // Free integrator memory :
  CVodeFree(&cvode_mem);
  N_VDestroy_Serial(yy) ; N_VDestroy_Serial(abstol) ; // heureusement, ne desallouent pas leurs 'double* NV_DATA_S()' car issus de 'N_VMake_Serial'
  if (rootfind != NULL) delete [] rootsfound ;
  if (nbVar_dot) {
	  for (j = 0 ; j < nbVar_dot ; j ++) 	 delete Var_primitive_sav[j] ;
	delete[] Var_primitive_sav ;
  }
//  _strtime_s(message, 100); cout << "at " << message << " :  exiting solver" << endl ; Update_Output();
	// strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  exiting solver" << endl ; 
	Update_Output();

	return 0 ;
}
/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */
	int check_flag(void *flagvalue, string funcname_, int opt) {
  int *errflag;
  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  const char * funcname = funcname_.c_str();
  if (opt == 0 && flagvalue == NULL) {
    (void)sprintf(message, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n", funcname); _LogMessage(message) ;
//    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
    return(1);
  }
  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      (void)sprintf(message, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n", funcname, *errflag); _LogMessage(message) ;
  //    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n", funcname, *errflag);
      return(1);
    }
  }
  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    (void)sprintf(message, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n", funcname); _LogMessage(message) ;
//    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n", funcname);
    return(1);
  }
  return(0);
}


int arkode(void(*f)(double, double*, double*), Fortran_vector& y, Fortran_vector &T, void(*aux)(double, double*), Fortran_vector& atol, Fortran_vector& rtol,
	int nbVar_dot, Fortran_vector** Var_primitive, Fortran_vector** Var_dot, bool verbose,	void(*rootfind)(double, double*, double*), int nrootfns) {
	ff = f;
	int itol = 1;
	/* 	itol = 1 (= CV_SS) : atol and rtol tous 2 scalaires ;
	itol = 2 (= CV_SV) : atol vecteur, rtol scalaire ;
	itol = 3 (= CV_WF) : ewt[i] defini par une fonction utilisateur (non encore implemente ; cf. doc CVODE p.29) */
	if (atol.size() > 1) itol++;
	if (rtol.size() > 1) itol += 2;
	if (itol == 3) { _LogMessage("Erreur itol = CV_WF : cas non encore traite par cvode"); return -1; }
	if (itol == 4) { _LogMessage("Erreur itol = 4 (atol et rtol tous deux vecteurs): non traite par cvode"); return -1; }
	//int neq = y.size();							 // taille du probleme = nb d'equations = nb de variables y[i]
	int nbt = T.size();							 // nombre de temps ou l'on veut la solution, y compris t0 = T[1]
	int i, j, flag, flagr;
	double t, tout, t_sav, dt;
	N_Vector abstol(InPlace_NVector(atol));
	N_Vector yy(InPlace_NVector(y));              // "habille" le Fortran_vector y en N_Vector sans occuper de memoire suppl.
	double* y_ = InPlace_Array(y);	// y_ = y.v_
	t = t_sav = T[1];
	Fortran_vector** Var_primitive_sav;
	if (aux != NULL) aux(T[1], y_); // instant t0 = T[1]
	Update_Output(true);
	if (nbVar_dot) {
		assert(Var_primitive);
		assert(Var_dot);
		for (j = 0; j < nbVar_dot; j++) {
			assert(Var_primitive[j]);
			assert(Var_dot[j]);
		}
		assert(Var_primitive[nbVar_dot] == NULL); assert(Var_dot[nbVar_dot] == NULL); // previendra l'utilisateur si on a declare moins de derivees que l'on n'en a dimensionnees
		Var_primitive_sav = new Fortran_vector*[nbVar_dot];
		for (j = 0; j < nbVar_dot; j++) 	  Var_primitive_sav[j] = new Fortran_vector(Var_primitive[j]->size());
	}
	arkode_mem = ERKStepCreate(ffff, T[1], yy);    // init. le solveur
	if (check_flag(arkode_mem, "ERKStepCreate", 0)) { _LogMessage("erreur ERKStepCreate"); return -1; }

	/* Call CVodeSVtolerances to specify the scalar relative tolerance
	* and vector absolute tolerances */
	flag = ERKStepSVtolerances(arkode_mem, rtol[1], abstol);
	if (check_flag(&flag, "ERKStepSVtolerances", 1)) return(1);
	int* rootsfound = NULL;
	if (rootfind != NULL) {
		if (nrootfns < 1) { _LogMessage("erreur RootFind : nrootfns =< 0 eq. a resoudre !"); return -1; }
		rootsfound = new int[nrootfns];
		flag = ERKStepRootInit(arkode_mem, nrootfns, gg);
		if (check_flag(&flag, "ERKStepRootInit", 1)) { _LogMessage("erreur ERKStepRootInit"); return -1; }
	}
	for (i = 2; i <= nbt; i++) {						// pour chaque instant ou l'on souhaite la solution
		tout = T[i];
		if (verbose) {
			strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  starting step n#" << i - 1 << " (tf = " << tout << ")" << endl;
			Update_Output(i == 2);
		}
		while (t < tout) {
			flag = -9999; // bidon
			while (flag != ARK_SUCCESS) {
				flag = ERKStepEvolve(arkode_mem, tout, yy, &t, ARK_ONE_STEP); // appel au solveur en mode 'ARK_ONE_STEP' // attention : cette instruction MAJ  t, qui peut donc maintenant etre > tout !!
				if (flag == ARK_ROOT_RETURN) {
					flagr = ERKStepGetRootInfo(arkode_mem, rootsfound);
					if (check_flag(&flagr, "ERKStepGetRootInfo", 1)) {
						_LogMessage(" Erreur ERKStepGetRootInfo");
				        //strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  exiting solver" << endl; 
						Update_Output(); return i;
					}
					for (j = 0; j < nrootfns; j++) {
						if (rootsfound[j] == 1) {
							(void)sprintf(message, "Eq.[%d] : Root Found at t = %g :", j + 1, t); _LogMessage(message);
							Update_Output();
						}
					}
					aux(t, y_);
				}
				else {
					if (flag != ARK_SUCCESS) {
						//    if (check_flag(&flag, "CVode", 1)) break ; // routine a  reprendre...
						(void)sprintf(message, "error-flag Arkode = %d", flag); _LogMessage(message);
						//strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  exiting solver" << endl; 
						Update_Output(true);
						return i;
					}
				}
			}
			if (nbVar_dot > 0) {
				dt = t - t_sav;
				for (j = 0; j < nbVar_dot; j++) {
					Var_dot[j]->set((*(Var_primitive[j]) - *(Var_primitive_sav[j])) / dt);
					Var_primitive_sav[j]->set(*(Var_primitive[j]));
				}
				if (t < tout) 	t_sav = t;
			}
		} // fin boucle  while(t < tout) : les 8 lignes suivantes sont donc executees si (t > tout) :
		if (aux != NULL) {		// calculs auxiliaires, par ex. sauvegarder var. intermediaires
								// a ce stade t_curr est juste superieur a tout : on interpole y a y_out = y(tout) :
			flag = ERKStepGetDky(arkode_mem, tout, 0, yy);
			check_flag(&flag, "ERKStepGetDky", 1);
			aux(T[i], y_);
		}
		t_sav = t;
	} // fin boucle i (T[i])
	if (verbose) {  // Print some final statistics :
		long int nst, nfe, netf;
		flag = ERKStepGetNumSteps(arkode_mem, &nst);
		check_flag(&flag, "ERKStepGetNumSteps", 1);
		flag = ERKStepGetNumRhsEvals(arkode_mem, &nfe);
		check_flag(&flag, "ERKStepGetNumRhsEvals", 1);
		flag = ERKStepGetNumErrTestFails(arkode_mem, &netf);
		check_flag(&flag, "ERKStepGetNumErrTestFails", 1);
		_LogMessage("\nFinal Statistics:");
		(void)sprintf(message, "nst = %-6ld nfe  = %-6ld netf = %-6ld\n", nst, nfe, netf);
		_LogMessage(message);
	}
	/* Free integrator memory : */
	ERKStepFree(&arkode_mem);
	N_VDestroy_Serial(yy); N_VDestroy_Serial(abstol); // heureusement, ne desallouent pas leurs 'double* NV_DATA_S()' car issus de 'N_VMake_Serial'
	if (rootfind != NULL) delete[] rootsfound;
	if (nbVar_dot) {
		for (j = 0; j < nbVar_dot; j++) 	 delete Var_primitive_sav[j];
		delete[] Var_primitive_sav;
	}
//	_strtime_s(message, 100); cout << "at " << message << " :  exiting solver" << endl; Update_Output();
	//strftime(message, 50, "%H:%M:%S", localtime(&current)) ; cout <<  "at " << message << " :  exiting solver" << endl; 
	Update_Output();
	return 0;
}


