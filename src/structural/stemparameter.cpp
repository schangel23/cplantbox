// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#include "stemparameter.h"

#include "Organism.h"
#include "Seed.h"
#include "tropism.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <assert.h>
#include <numeric>

namespace CPlantBox {

/**
 * @return Mean maximal stem length of this stem type
 */
double StemSpecificParameter::getK() const {
    double l = std::accumulate(ln.begin(), ln.end(), 0.);
    return l+la+lb;
}

/**
 * @copydoc OrganParameter::toString()
 */
std::string StemSpecificParameter::toString() const
{
    std::stringstream str;
    str << "subType\t" << subType << std::endl;
    str << "lb\t" << lb << std::endl << "la\t" << la << std::endl;
    str << "nob\t" << nob() << std::endl << "r\t" << r << std::endl << "a\t" << a << std::endl;
    str << "theta\t" << theta << std::endl << "rlt\t" << rlt << std::endl;
    str << "ln\t";
    for (int i=0; i<ln.size(); i++) {
        str << ln[i] << " ";
    }
    str << std::endl;
    return str.str();
}



/**
 * Default constructor sets up hashmaps for class introspection
 */
StemRandomParameter::StemRandomParameter(std::shared_ptr<Organism> plant) :OrganRandomParameter(plant)
{
    // base class default values
    name = "undefined";
    organType = Organism::ot_stem;
    subType = -1;
    f_tf = std::make_shared<Tropism>(plant);
    bindParameters();
}

/**
 * @copydoc OrganTypeParameter::copy()
 */
std::shared_ptr<OrganRandomParameter> StemRandomParameter::copy(std::shared_ptr<Organism> plant)
{
    auto r = std::make_shared<StemRandomParameter>(*this); // copy constructor breaks class introspection
    r->plant = plant;
    r->bindParameters(); // fix class introspection
    r->f_tf = f_tf->copy(plant); // copy call back classes
    r->f_gf = f_gf->copy();
    r->f_se = f_se->copy();
    r->f_sa = f_sa->copy();
    r->f_sbp = f_sbp->copy();
    return r;
}

/**
 * @copydoc OrganTypeParameter::realize()
 *
 * Creates a specific stem from the stem type parameters.
 * @return Specific stem parameters derived from the stem type parameters
 */
std::shared_ptr<OrganSpecificParameter> StemRandomParameter::realize()
{
    auto p = plant.lock();
	double lb_;
    double la_;
    std::vector<double> ln_; // stores the inter-distances
	double res;
    double nob_sd = p->randn()*nobs();
    int nob_real = round(std::max(nob() + nob_sd, 0.)); // real maximal number of branching points
	bool hasLaterals = (successorST.size()>0) ;
	if (dx <= dxMin){
		std::cout<<"dx <= dxMin, dxMin set to dx/2"<<std::endl;
		this->dxMin = dx/2;
	}
	if (!hasLaterals) { // no laterals

    	lb_ = 0;
        la_ = std::max(lmax + p->randn()*lmaxs, 0.); // la, and lb is ignored
		res = la_-floor(la_ / dx)*dx;
		if(res < dxMin && res != 0){
			if(res <= dxMin/2){ la_ -= res;
			}else{la_ =  floor(la_ / dx)*dx + dxMin;}
		}			//make la_ compatible with dx() and dxMin()

    } else {
    lb_ = std::max(lb + p->randn()*lbs, 0.); // length of basal zone
	la_ = std::max(la + p->randn()*las, 0.); // length of apical zone
	res = lb_ - floor(lb_/dx)* dx;
	if((res < dxMin) && (res != 0)){
		if(res <= dxMin/2){ lb_ -= res;
		}else{lb_ =  floor(lb_ / dx)*dx + dxMin;}
	}

	bool hasSeed = (p->baseOrgans.size()>0)&&(p->baseOrgans.at(0)->organType()==Organism::ot_seed);
	if(hasSeed&&(lb_< dxMin*2)&&(p->getSeed()->param()->nC >0))//lb must be longer than nZ. TODO:remove when root laterals is implemented
	{
		lb_ = dxMin*2;
	}

    res = la_-floor(la_ / dx)*dx;
	if(res < dxMin && res != 0){
		if(res <= dxMin/2){ la_ -= res;
		}else{la_ =  floor(la_ / dx)*dx + dxMin;}
	}
	double ln_mean = ln;
	if(ln < dxMin*0.99 && ln !=0){
		std::cout<<"\nStemRandomParameter::realize inter-lateral distance (ln) "<<ln<<" below minimum resolution (dxMin) "<<dxMin<<". ln set to dxMin"<<std::endl;
		ln_mean = dxMin;
	}

	//adapt number of laterals by branching point to keep same total number of lats
	//in spite of dxMin
	int nob1 = std::max((lmax-la_-lb_)/ln_mean+1, 1.);//use new la_, lb_ and ln_mean
    int nob_ = std::min(std::max(round(nob1 + p->randn()*nobs()), 1.),double(nob_real));// maximal number of branches
	int latMissing = nob_real - nob_;
	int latExtra1 = floor(latMissing/nob_);//mean number of extra laterals per branching point to keep correct number
	int latExtra2 = latMissing - latExtra1*(nob_);
	int latExtra2_ = latExtra2;

		//at end of basal zone
		for (int j = 0; j<latExtra1; j++) { ln_.push_back(0);}
		if (latExtra2_> 0) {ln_.push_back(0);latExtra2_--;}

		switch(lnf) {
		case 0: // homogeneously distributed stem nodes
		for (int i = 0; i<nob_-1; i++) { // create inter-stem distances
			double d = std::max(ln_mean +p->randn()*lns,1.e-5); //Normal function of equal internode distance
			res = d -floor(d / dx)*dx;
			if(res < dxMin && res != 0){
				if(res <= dxMin/2){d -= res;
				}else{d = floor(d / dx)*dx + dxMin;}

				} //make ln compatible with dx() and dxMin().

			ln_.push_back(d);
			for (int j = 0; j<latExtra1; j++) { ln_.push_back(0);}
			if (latExtra2_> 0) {ln_.push_back(0);latExtra2_--;}


		};break;
		case 1: //  nodes distance increase linearly
		for (int i = 0; i<nob_*2-1; i++) { // create inter-stem distances
			double d =  std::max(ln_mean*(1+i) +p->randn()*lns,1.e-5); //std::max(  );//ln +p->randn()*lns,1e-9);
			res = d -floor(d / dx)*dx;
			if(res < dxMin && res != 0){
				if(res <= dxMin/2){d -= res;
				}else{d = floor(d / dx)*dx + dxMin;}

				} //make ln compatible with dx() and dxMin().

			ln_.push_back(d);
			for (int j = 0; j<latExtra1; j++) { ln_.push_back(0);}
			if (latExtra2_> 0) {ln_.push_back(0);latExtra2_-=0.5;}
			ln_.push_back(0);
			for (int j = 0; j<latExtra1; j++) { ln_.push_back(0);}
			if (latExtra2_> 0) {ln_.push_back(0);latExtra2_-=0.5;}

		};break;
		case 2: //nodes distance decrease linearly
		for (int i = 0; i<nob_-1; i++) { // create inter-stem distances
			double d =  std::max(ln_mean*(1+i) +p->randn()*lns,1.e-5); //std::max(  );//ln +p->randn()*lns,1e-9);
			res = d -floor(d / dx)*dx;
			if(res < dxMin && res != 0){
				if(res <= dxMin/2){d -= res;
				}else{d = floor(d / dx)*dx + dxMin;}

				} //make ln compatible with dx() and dxMin().

			ln_.push_back(d);
			for (int j = 0; j<latExtra1; j++) { ln_.push_back(0);}
			if (latExtra2_> 0) {ln_.push_back(0);latExtra2_--;}

		};break;
		case 3: //nodes distance decrease exponential
		for (int i = 0; i<nob_-1; i++) { // create inter-stem distances
			double d =  std::max(ln_mean +p->randn()*lns,1.e-5); //std::max(  );//ln +p->randn()*lns,1e-9);
			res = d -floor(d / dx)*dx;
			if(res < dxMin && res != 0){
				if(res <= dxMin/2){d -= res;
				}else{d = floor(d / dx)*dx + dxMin;}

				} //make ln compatible with dx() and dxMin().

			ln_.push_back(d);
			for (int j = 0; j<latExtra1; j++) { ln_.push_back(0);}
			if (latExtra2_> 0) {ln_.push_back(0);latExtra2_--;}

		};break;

		case 4://nodes distance decrease exponential
		for (int i = 0; i<nob_*2-1; i++) { // create inter-stem distances
			double d =  std::max(ln_mean/(1+i) +p->randn()*lns,1.e-5); //std::max(  );//ln +p->randn()*lns,1e-9);
			res = d -floor(d / dx)*dx;
			if(res < dxMin && res != 0){
				if(res <= dxMin/2){d -= res;
				}else{d = floor(d / dx)*dx + dxMin;}

				} //make ln compatible with dx() and dxMin().

			ln_.push_back(d);
			for (int j = 0; j<latExtra1; j++) { ln_.push_back(0);}
			if (latExtra2_> 0) {ln_.push_back(0);latExtra2_-=0.5;}
			ln_.push_back(0);
			for (int j = 0; j<latExtra1; j++) { ln_.push_back(0);}
			if (latExtra2_> 0) {ln_.push_back(0);latExtra2_-=0.5;}
		}; break;
		case 5://nodes distance decrease exponential
		for (int i = 0; i<nob_*2-1; i++) { // create inter-stem distances
			double d =  std::max(ln_mean/(1+i) +p->randn()*lns,1.e-5); //std::max(  );//ln +p->randn()*lns,1e-9);
			res = d -floor(d / dx)*dx;
			if(res < dxMin && res != 0){
				if(res <= dxMin/2){d -= res;
				}else{d = floor(d / dx)*dx + dxMin;}

				} //make ln compatible with dx() and dxMin().

			ln_.push_back(d);
			for (int j = 0; j<latExtra1; j++) { ln_.push_back(0);}
			if (latExtra2_> 0) {ln_.push_back(0);latExtra2_--;}
		};break;
default:
		throw std::runtime_error("StemRandomParameter::realize type of inter-branching distance not recognized");
}}
    double r_ = std::max(r + p->randn()*rs, 0.); // initial elongation
    double a_ = std::max(a + p->randn()*as, 0.); // radius
    double theta_ = std::max(theta + p->randn()*thetas, 0.); // initial elongation
    double rlt_ = std::max(rlt + p->randn()*rlts, 0.); // stem life time
	double delayNGStart_ = std::max(delayNGStart + p->randn()*delayNGStarts, 0.);
	double delayNGEnd_ = std::max(delayNGEnd + p->randn()*delayNGEnds, 0.);
	if(delayNGEnd_ < delayNGStart_){
		std::cout<<"StemRandomParameter::realize() : delayNGEnd_ < delayNGStart_ \n";
		std::cout<<"set delayNGEnd_ = delayNGStart_ = "<<delayNGStart_<<std::endl;
		delayNGEnd_ = delayNGStart_;
	}
	double ldelay_ = std::max(ldelay + p->randn()*ldelays, 0.);

    // ---------------------------------------------------------------
    // Fournier-Andrieu (FA) override of the realised inter-lateral
    // distance vector ``ln_``.  Plan B.2 (peduncle exuberance fix,
    // 2026-04-27).  When the FA flag is on we replace the
    // ``lmax/ln_mean``-derived sizing+filling above with an
    // ``successorST.size()``-derived sizing and a per-rank
    // ``internode_IL_final`` filling.  This:
    //   (a) eliminates the size-19 phantom that overshoots the 17
    //       phytomer slots ``successorWhere`` defines (16 leaves +
    //       1 tassel for maize_calibrated.xml), and
    //   (b) replaces the uniform ~10 cm scalar sampling by the
    //       per-rank Déa profile from ``phase_III_per_rank.json``.
    // FA-off path is bit-identical: this block is gated on the flag
    // and only fires for stems that explicitly opt in.  The basal
    // ``basal_zero_ranks`` set keeps ranks 1..4 at 0 so the basal
    // stub seeded by Stem::simulate's plastochron loop is the only
    // length contribution there.  No RNG pulls in this branch.
    if (this->use_fournier_andrieu_kinetics
        && hasLaterals
        && this->successorST.size() > 0
        && this->internode_IL_final.size() > 0) {
        // CPlantBox convention (Stem.cpp:508 + plastochron loop): the
        // expression ``p.ln.size() + 1`` is treated as the maximum number
        // of laterals.  With N successor rules (= N attachable laterals),
        // ln must have exactly N - 1 inter-lateral entries; otherwise the
        // S3b.7 plastochron loop iterates past the last actual successor
        // (n = N + 1) and the topmost-lateral basal_step gate
        // ``if (n < n_laterals_max)`` fires once spuriously on the tassel
        // attach, producing the 1 cm phantom advance reported in the
        // codex-rescue Finding 1 (2026-04-27).  Sizing ``ln_fa`` to
        // ``successorST.size() - 1`` aligns the FA path with the
        // existing convention.
        const std::size_t n_lats = this->successorST.size();
        const std::size_t n_phytomers = (n_lats > 0) ? (n_lats - 1) : 0;
        std::vector<double> ln_fa(n_phytomers, 0.0);
        const std::size_t il_n = this->internode_IL_final.size();
        const auto& bz = this->basal_zero_ranks;
        auto is_basal_zero = [&](int rank_one_indexed) -> bool {
            return std::find(bz.begin(), bz.end(), rank_one_indexed) != bz.end();
        };
        // Match Stem::simulate's fa_sum semantics exactly: each rank's
        // length_per_n[n] is seeded to basal_internode_cm by the S3b.7
        // plastochron loop, then driven by FA kinetics toward IL_final.
        // fa_sum[n] = max(target_n, length_per_n[n]) ≥ basal_internode_cm
        // for every initiated rank.  Aligning ln[i] with the same floor
        // keeps the branching-zone cap (sum(ln)) equal to fa_sum at
        // maturity, so targetlength never exceeds the realisable
        // branching length and the apical-zone block stops absorbing
        // residual dl.  Plan B.3 D.5 (mainstem-top bound) closes here.
        const double basal_floor = std::max(0.0, this->basal_internode_cm);
        auto is_leaf_at = [&](std::size_t idx) -> bool {
            if (idx >= this->successorOT.size()) return false;
            for (int ot : this->successorOT.at(idx)) {
                if (ot == Organism::ot_leaf) return true;
            }
            return false;
        };
        for (std::size_t i = 0; i < n_phytomers; ++i) {
            const int rank = static_cast<int>(i) + 1; // 1-indexed
            // Defensive: under the new sizing ``i`` should always index a
            // leaf successor (the tassel lives at position n_lats - 1 and
            // is handled implicitly via the next_is_tassel gate below).
            // Keep this branch as a safety net for unusual successor
            // patterns (e.g. interior non-leaf successor) so we don't
            // emit IL_final into a non-leaf slot.
            const bool is_leaf_successor = is_leaf_at(i);
            if (!is_leaf_successor) {
                ln_fa[i] = 0.0;
                continue;
            }
            // Plan B.3 (peduncle exuberance, 2026-04-27) HI#4 gate +
            // codex-rescue Finding 3 (2026-04-27): collapse this mainstem
            // entry to basal_floor only when ALL of:
            //   (a) the topmost (last) successor in the full table is
            //       non-leaf — i.e., the plant ends in a tassel/spike,
            //       not a leaf;
            //   (b) we are the slot immediately below it (i + 1 ==
            //       n_lats - 1).
            // This restricts the peduncle collapse to the maize tassel
            // pattern (and equivalents).  An interior non-leaf successor
            // (e.g., a branched stem inserted between leaves on a
            // hypothetical FA-on wheat) is left on the standard
            // IL_final / basal_floor path, so its preceding internode
            // is not silently truncated.
            const bool topmost_is_non_leaf =
                (n_lats > 0) && !is_leaf_at(n_lats - 1);
            const bool is_peduncle_slot =
                topmost_is_non_leaf && (i + 1 == n_lats - 1);
            if (is_peduncle_slot) {
                ln_fa[i] = basal_floor;
                continue;
            }
            // Basal_zero ranks: only the basal_step seed contributes (no
            // FA elongation).  Setting ln to the basal floor keeps
            // internodalGrowth's per-phytomer cap consistent with the
            // seeded geometry, while basal_zero_ranks gate at line ~822
            // still pins growth to 0.
            if (is_basal_zero(rank)) {
                ln_fa[i] = basal_floor;
                continue;
            }
            const std::size_t il_idx = static_cast<std::size_t>(rank - 1);
            const double il = (il_idx < il_n) ? this->internode_IL_final.at(il_idx) : 0.0;
            // Floor to basal_internode_cm so the branching-zone cap covers
            // the seeded basal_step on every rank (rank 5 has IL_final=0.8
            // for Déa, basal_step=1.0 for maize_calibrated → floor=1.0).
            ln_fa[i] = std::max(il, basal_floor);
        }
        ln_ = std::move(ln_fa);
    }


    auto sp = std::make_shared<StemSpecificParameter>(subType,lb_,la_,ln_,r_,a_,theta_,rlt_,hasLaterals, this->nodalGrowth, delayNGStart_, delayNGEnd_, ldelay_);

    // Fournier-Andrieu kinetics pass-through (no RNG pulls — Hard Invariant #5
    // preserved: flag and vectors copy as literals; when flag=false the
    // downstream Stem::simulate path never consults these fields).
    sp->use_fournier_andrieu_kinetics = this->use_fournier_andrieu_kinetics;
    sp->internode_v_n = this->internode_v_n;
    sp->internode_D_n = this->internode_D_n;
    sp->internode_IL_final = this->internode_IL_final;

    // Genotypic FA-asymptote scale factor H (PLAN_CULTIVAR_HEIGHT_FACTOR_2026-05-07
    // §S1). Gate the randn() pull on cultivar_height_factor_s > 0 so the RNG
    // state is bit-identical to pre-S1 HEAD when an XML omits the field
    // (D3 / D.0 6-XML invariance). With default _s=0.0 we skip randn()
    // entirely and write H=cultivar_height_factor literally. The 0.1 floor
    // prevents pathologically small / negative draws under wide _s.
    double H_draw = this->cultivar_height_factor;
    if (this->cultivar_height_factor_s > 0.0) {
        H_draw = std::max(0.1, this->cultivar_height_factor
                               + p->randn() * this->cultivar_height_factor_s);
    }
    sp->cultivar_height_factor = H_draw;
    return sp;
}


/**
 * todo docme
 *
 * todo I have no idea why this holds...
 */
double StemRandomParameter::nobs() const
{
	double nobs = 0;
	if(ln >0)
	{
		double nobs = (lmaxs/lmax - lns/ln)*lmax/ln; // error propagation
		if (la>0) {
			nobs -= (las/la - lns/ln)*la/ln;
		}
		if (lb>0) {
			nobs -= (lbs/lb - lns/ln)*lb/ln;
		}
	}
    return std::max(nobs,0.);
}




/**
 * Sets up class introspection by linking parameter names to their class members,
 * additionally adds a description for each parameter, for toString and writeXML
 */
void StemRandomParameter::bindParameters()
{
    OrganRandomParameter::bindParameters();
    bindParameter("lb", &lb, "Basal zone [cm]", &lbs);
    bindParameter("la", &la, "Apical zone [cm]", &las);
    bindParameter("ln", &ln, "Inter-lateral distance [cm]", &lns);
    bindParameter("lnf", &lnf, "Type of inter-branching distance (0 homogeneous, 1 linear inc, 2 linear dec, 3 exp inc, 4 exp dec)");
    bindParameter("lmax", &lmax, "Maximal stem length [cm]", &lmaxs);
    bindParameter("r", &r, "Initial growth rate [cm day-1]", &rs);
    bindParameter("a", &a, "Stem radius [cm]", &as);
    bindParameter("RotBeta", &rotBeta, "RevRotation of the stem");  /// todo improve description, start lower letter
    bindParameter("BetaDev", &betaDev, "RevRotation deviation");  /// todo improve description, start lower letter
    bindParameter("InitBeta", &initBeta, "Initial RevRotation");  /// todo improve description, start lower letter
    bindParameter("tropismT", &tropismT, "Type of stem tropism (plagio = 0, gravi = 1, exo = 2, hydro, chemo = 3)");
    bindParameter("tropismN", &tropismN, "Number of trials of stem tropism");
    bindParameter("tropismS", &tropismS, "Mean value of expected change of stem tropism [1/cm]");
	bindParameter("tropismAge", &tropismAge, "Age at which organ switch tropism", &tropismAges);
    bindParameter("theta", &theta, "Angle between stem and parent stem [rad]", &thetas);
    bindParameter("rlt", &rlt, "Stem life time [day]", &rlts);
    bindParameter("gf", &gf, "Growth function number [1]", &rlts);
	bindParameter("nodalGrowth", &nodalGrowth, "nodal growth function (sequential = 0, equal = 0)");
    bindParameter("delayNGStart", &delayNGStart, "delay between stem creation and start of nodal growth", &delayNGStarts);
    bindParameter("delayNGEnd", &delayNGEnd, "delay between stem creation and start of nodal growth", &delayNGEnds);
    bindAxisParameter("delayNGEnd", &delayNGEndAxis); ///< Lock #1: axis="TT" → MultiPhaseStemGrowth interprets delayNGEnd as Andrieu-TT cessation threshold
    bindParameter("ldelay", &ldelay, "delay between latteral creation and start of nodal growth", &ldelays);
    bindParameter("use_thermal_emergence", &use_thermal_emergence, "Use thermal-time gated emergence [0/1]");
    bindParameter("tt_emergence", &tt_emergence, "Thermal-time emergence threshold [degCd], <0 disables");
    bindParameter("use_thermal_cessation", &use_thermal_cessation, "Use thermal-time cessation (freezes nodal growth at VT) [0/1]");
    bindParameter("tt_cessation", &tt_cessation, "Thermal-time cessation threshold [degCd], <0 disables");
    bindParameter("plastochron_andrieu", &plastochron_andrieu, "Plastochron on Andrieu Tb=9.8 axis [degCd/rank] (FA 2000 Déa ~23)");
    bindParameter("basal_internode_cm", &basal_internode_cm, "Fixed internode spacing for basal_zero_ranks [cm]");
    bindParameter("use_fournier_andrieu_kinetics", &use_fournier_andrieu_kinetics, "Use Fournier-Andrieu per-phytomer internode kinetics [0/1]"); ///< S0.7 (Lock #3 Half A): bind so the FA flag survives writeParameters → readXML round-trip; required for D6's pure-XML invocation contract.
    bindParameter("phase_IV_k", &phase_IV_k, "FA Phase IV exponential decay rate [1/degCd]; lower = slower asymptote, smoother post-cliff stem ramp (FA 2000 default 0.09)");
    bindParameter("phase_I_duration", &phase_I_duration, "FA Phase I duration [degCd] (FA 2000 default 309)");
    bindParameter("phase_II_duration", &phase_II_duration, "FA Phase II duration [degCd] (FA 2000 default 25)");
    bindParameter("phase_IV_duration", &phase_IV_duration, "FA Phase IV operational duration [degCd] (FA 2000 default 30)");
    bindParameter("il_init_cm", &il_init_cm, "FA initial internode length at tau=0 [cm] (Zhu 2014 default 0.0025)");
    bindParameter("il_at_end_phase_II_cm", &il_at_end_phase_II_cm, "FA IL at end of Phase II [cm] (FA 2000 line 223 default 4.5)");
    bindParameter("half_plastochron_lag_degCd", &half_plastochron_lag_degCd, "FA lag between leaf primordium and internode init [degCd] (FA 2000 line 207 default 9.6)");
    bindParameter("collar_frac_of_dlin", &collar_frac_of_dlin, "FA α in collar_TT = T0 + lag_exp + α·D_lin (FA 2005 / AHB 2006 literal default 1.0)");
    bindParameter("cultivar_height_factor", &cultivar_height_factor,
                  "Genotypic FA Phase III/IV asymptote scale (default 1.0); active only with use_fournier_andrieu_kinetics=1",
                  &cultivar_height_factor_s);
}

/**
 * S0.6 — Read per-rank `MultiPhaseStemGrowth` arrays (`v_n`, `D_n`, `IL_final`)
 * and basal-zero ranks from XML, on top of the scalar fields handled by the
 * base reader. Format mirrors the leafGeometry/leafCurvature precedent in
 * `LeafRandomParameter::readXML` but uses a single comma-separated `values=`
 * attribute per array (the FA tables are short — 16-17 ranks — so per-element
 * tags would be needlessly verbose).
 *
 * Back-compat: an XML that omits these tags leaves the constructor defaults
 * untouched (`internode_*` empty; `basal_zero_ranks={1,2,3,4}`), so every
 * existing file (wheat, brassica, carbon2020, 2020-maize, modelparam_4,
 * maize_calibrated FA-off) loads bit-identically.
 */
void StemRandomParameter::readXML(tinyxml2::XMLElement* element, bool verbose)
{
    OrganRandomParameter::readXML(element, verbose);

    // Reset double-vector arrays before parsing so a re-read overwrites rather
    // than appending. Leave basal_zero_ranks alone unless the XML explicitly
    // overrides it (the constructor default {1,2,3,4} is the canonical maize
    // Zhu-2014/He-2021 convention; clobbering it would silently change every
    // FA-off stem on round-trip).
    internode_v_n.resize(0);
    internode_D_n.resize(0);
    internode_IL_final.resize(0);
    bool basal_zero_ranks_overridden = false;

    auto p = element->FirstChildElement("parameter");
    while (p) {
        const char* str = p->Attribute("name");
        if (str != nullptr) {
            std::string key = std::string(str);
            const char* values = p->Attribute("values");
            if (values != nullptr) {
                if (key == "v_n") {
                    internode_v_n = string2vector(values, 0.0);
                } else if (key == "D_n") {
                    internode_D_n = string2vector(values, 0.0);
                } else if (key == "IL_final") {
                    internode_IL_final = string2vector(values, 0.0);
                } else if (key == "basal_zero_ranks") {
                    if (!basal_zero_ranks_overridden) {
                        basal_zero_ranks.resize(0);
                        basal_zero_ranks_overridden = true;
                    }
                    auto v = string2vector(values, 0);
                    basal_zero_ranks.insert(basal_zero_ranks.end(), v.begin(), v.end());
                }
            }
        }
        p = p->NextSiblingElement("parameter");
    }
}

/**
 * S0.6 — Emit per-rank arrays alongside the scalar fields written by the
 * base writer. Empty arrays are skipped so non-FA stems round-trip without
 * gaining empty `<parameter name="v_n" values=""/>` tags. `basal_zero_ranks`
 * is gated on the FA flag so the constructor default {1,2,3,4} is not
 * sprinkled into every stem XML on every round-trip — only stems that
 * actually opt into FA kinetics need it persisted.
 */
tinyxml2::XMLElement* StemRandomParameter::writeXML(tinyxml2::XMLDocument& doc, bool comments) const
{
    tinyxml2::XMLElement* element = OrganRandomParameter::writeXML(doc, comments);

    auto append_array_double = [&](const char* name,
                                   const std::vector<double>& v,
                                   const char* descr) {
        if (v.empty()) return;
        tinyxml2::XMLElement* p = doc.NewElement("parameter");
        p->SetAttribute("name", name);
        p->SetAttribute("values", vector2string(v).c_str());
        element->InsertEndChild(p);
        if (comments && descr) {
            tinyxml2::XMLComment* c = doc.NewComment(descr);
            element->InsertEndChild(c);
        }
    };
    auto append_array_int = [&](const char* name,
                                const std::vector<int>& v,
                                const char* descr) {
        if (v.empty()) return;
        tinyxml2::XMLElement* p = doc.NewElement("parameter");
        p->SetAttribute("name", name);
        p->SetAttribute("values", vector2string(v).c_str());
        element->InsertEndChild(p);
        if (comments && descr) {
            tinyxml2::XMLComment* c = doc.NewComment(descr);
            element->InsertEndChild(c);
        }
    };

    append_array_double("v_n",      internode_v_n,
                        "MultiPhaseStemGrowth Phase III rate per rank [cm/degCd]");
    append_array_double("D_n",      internode_D_n,
                        "MultiPhaseStemGrowth Phase III duration per rank [degCd]");
    append_array_double("IL_final", internode_IL_final,
                        "MultiPhaseStemGrowth Phase IV asymptote per rank [cm]");
    if (use_fournier_andrieu_kinetics) {
        append_array_int("basal_zero_ranks", basal_zero_ranks,
                         "Ranks pinned to IL_final=0 (basal-zero, Zhu 2014 / He 2021)");
    }
    return element;
}

} // end namespace CPlantBox
