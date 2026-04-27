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
        const std::size_t n_phytomers = this->successorST.size();
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
            // Tassel/non-leaf successor rule: peduncle elongation belongs
            // to the tassel subType, not the mainstem; this slot stays 0.
            const bool is_leaf_successor = is_leaf_at(i);
            if (!is_leaf_successor) {
                ln_fa[i] = 0.0;
                continue;
            }
            // Plan B.3 (peduncle exuberance, 2026-04-27) HI#4 gate: if the
            // NEXT successor slot is non-leaf (tassel), ln[i] is the
            // mainstem internode immediately below the tassel attachment
            // — i.e., the peduncle.  Hand the peduncle to the tassel
            // subType (whose own lb/internode handles it) by collapsing
            // this mainstem entry to basal_floor.  Without this gate, the
            // mainstem skeleton extends ~17 cm above the topmost leaf
            // (matching IL_final[16] for Déa) and HI#4 (mainstem top ≤
            // topmost-leaf insertion + 5 cm) cannot close.  The lofter
            // cosmetic-trim removal in S4 depends on this collapse.
            const bool next_is_tassel = (i + 1 < n_phytomers) && !is_leaf_at(i + 1);
            if (next_is_tassel) {
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
    bindParameter("ldelay", &ldelay, "delay between latteral creation and start of nodal growth", &ldelays);
    bindParameter("use_thermal_emergence", &use_thermal_emergence, "Use thermal-time gated emergence [0/1]");
    bindParameter("tt_emergence", &tt_emergence, "Thermal-time emergence threshold [degCd], <0 disables");
    bindParameter("use_thermal_cessation", &use_thermal_cessation, "Use thermal-time cessation (freezes nodal growth at VT) [0/1]");
    bindParameter("tt_cessation", &tt_cessation, "Thermal-time cessation threshold [degCd], <0 disables");
    bindParameter("plastochron_andrieu", &plastochron_andrieu, "Plastochron on Andrieu Tb=9.8 axis [degCd/rank] (FA 2000 Déa ~23)");
    bindParameter("basal_internode_cm", &basal_internode_cm, "Fixed internode spacing for basal_zero_ranks [cm]");
}

} // end namespace CPlantBox
