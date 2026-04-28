// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#ifndef STEMPARAMETER_H_
#define STEMPARAMETER_H_

#include "mymath.h"
#include "soil.h"
#include "growth.h"
#include "organparameter.h"

/**
 * This file describes the classes StemSpecificParameter and StemRandomParameter.
 * StemSpecificParameter are drawn from the StemRandomParameter class
 */

namespace CPlantBox {

class Organism;

/**
 * Parameters of a specific stem, its created by StemRandomParameter:realize()
 */
class StemSpecificParameter :public OrganSpecificParameter
{

public:

    StemSpecificParameter(): StemSpecificParameter(-1,0.,0.,std::vector<double>(0),0,0.,0.,0.,0.) { } ///< Default constructor
    StemSpecificParameter(int type, double lb, double la, const std::vector<double>& ln, double r, double a, double theta, double rlt,
	bool laterals= false, int nodalGrowth = 0, double delayNGStart = 0.,double delayNGEnd = 0., double delayLat = 0.):
        OrganSpecificParameter(type, a),  lb(lb), la(la), r(r), theta(theta), rlt(rlt), ln(ln),
		laterals(laterals), nodalGrowth(nodalGrowth), delayNGStart(delayNGStart), delayNGEnd(delayNGEnd){ } ///< Constructor setting all parameters

    /* Fournier-Andrieu per-phytomer internode kinetics (opt-in; see plan
     * PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §B.1).
     * Default 0 on every constructor path — Hard Invariant #5 preserved.
     * S0.7 (Lock #3 Half A): typed `int` (not `bool`) so it survives
     * `OrganRandomParameter::bindParameter` round-trip via XML; matches the
     * `use_fa_kinetics` / `use_thermal_*` flag pattern on other parameter
     * classes. Truthy semantics unchanged for existing call sites. */
    int use_fournier_andrieu_kinetics = 0;
    std::vector<double> internode_v_n;       ///< per-rank Phase III rate [cm/degCd], empty unless flag is true
    std::vector<double> internode_D_n;       ///< per-rank Phase III duration [degCd], empty unless flag is true
    std::vector<double> internode_IL_final;  ///< per-rank Phase IV asymptote [cm], empty unless flag is true (indexed 1-based by rank)

    /*
     * Stem parameters per single stem
     */
    double lb;              ///< Basal zone [cm]
    double la;              ///< Apical zone [cm]
    double r;               ///< Initial growth rate [cm day-1]
    double theta;           ///< Angle between stem and parent stem [rad]
    double rlt;             ///< Stem life time [day]
    std::vector<double> ln; ///< Inter-lateral distances [cm]

    bool laterals = false;
	int nodalGrowth;			///< whether to implement the internodal growth [1] (see @stem::simulate)
	double delayNGStart;
	double delayNGEnd;

    int nob() const { return ln.size() + laterals; } ///< return the maximal number of branching points [1]
    double getK() const; ///< Returns the exact maximal stem length of this realization [cm]

    std::string toString() const override; ///< for debugging

};



/**
 * Contains a parameter set describing a stem type
 */
class StemRandomParameter :public OrganRandomParameter
{

public:

    StemRandomParameter(std::shared_ptr<Organism> plant); ///< default constructor
    virtual ~StemRandomParameter() { };

    std::shared_ptr<OrganRandomParameter> copy(std::shared_ptr<Organism> plant) override;

    std::shared_ptr<OrganSpecificParameter> realize() override; ///< Creates a specific stem from the stem parameter set

    /* S0.6: per-rank Phase III/IV arrays (v_n, D_n, IL_final) and basal_zero_ranks
     * are vector fields that the scalar-only base reader cannot handle. Override
     * read/write to parse/emit `<parameter name="..." values="..."/>` tags.
     * Existing XMLs without these tags load identically (arrays remain empty;
     * basal_zero_ranks keeps its constructor default {1,2,3,4}).               */
    void readXML(tinyxml2::XMLElement* element, bool verbose) override;
    tinyxml2::XMLElement* writeXML(tinyxml2::XMLDocument& doc, bool comments = true) const override;

    double nob() const { if(ln>0){ return std::max((lmax-la-lb)/ln+1, 1.);}else{return 1.;} }  ///< returns the mean maximal number of branching nodes [1]
    double nobs() const; ///< returns the standard deviation of number of branches [1]


    /*
     * Parameters per stem typedelayNGStart
     */
    double lb = 0.; 	    ///< Basal zone [cm]
    double lbs = 0.;        ///< Standard deviation basal zone [cm]
    double la = 0.;	        ///< Apical zone [cm];
    double las = 0.;    	///< Standard deviation apical zone [cm];
    double ln = 1; 		    ///< Inter-lateral distance [cm]
    double lns = 0.;    	///< Standard deviation inter-lateral distance [cm]
    int lnf = 0;            ///< type of inter-branching distance (0 homogeneous, 1 linear inc, 2 linear dec, 3 exp inc, 4 exp dec)

    double lmax = 0.;       ///< Maximal stem length [cm]
    double lmaxs = 0.;      ///< Standard deviation of maximal stem length [cm]
    double r = 1;		    ///< Initial growth rate [cm day-1]
    double rs = 0.;	    	///< Standard deviation initial growth rate [cm day-1]
    double rotBeta = 0.6;	///< Revrotation
    double betaDev = 0.2;	///< Deviation of RevRotation
    double initBeta = 0.2;	///< Initial RevRotation
    int tropismT = 1;	    ///< Stem tropism parameter (Type) tt_plagio = 0, tt_gravi = 1, tt_exo = 2, tt_hydro = 3, tt_antigravi = 4, tt_twist = 5,  tt_antigravi2gravi = 6
    double tropismN = 1.;   ///< Stem tropism parameter (number of trials)
    double tropismS = 0.;  ///< Stem tropism parameter (mean value of expected changeg) [1/cm]
	double tropismAge = 0.;	///< Leaf tropism parameter (age when switch tropism)
	double tropismAges = 0.;///< Leaf tropism parameter (age when switch tropism, standard deviation)
    double theta = 0.; 	///< Angle between stem and parent stem (rad)
    double thetas= 0.; 	    ///< Standard deviation angle between stem and parent stem (rad)
    double rlt = 1e9;		///< Stem life time (days)
    double rlts = 0.;	    ///< Standard deviation stem life time (days)
    int gf = 1;			    ///< Growth function (1=negative exponential, 2=linear)
	int nodalGrowth = 1;		///< whether to implement the internodal growth (see @stem::simulate)
	double delayNGStart = 0.;		///< delay between stem creation and start of nodal growth [day]
	double delayNGStarts = 0.;		///< delay between stem creation and start of nodal growth, deviation [day]
	double delayNGEnd = 0.;		///< delay between stem creation and start of nodal growth [day]
	double delayNGEnds = 0.;		///< delay between stem creation and start of nodal growth, deviation [day]

    /* Thermal-time emergence gate (mirror of LeafRandomParameter fields).
     * When use_thermal_emergence=1, the stem's growth is gated on plant accumulated TT
     * reaching tt_emergence [degCd]. Used e.g. for tassel subType to emerge at VT
     * under variable temperature forcing. tt_emergence<0 disables. */
    int use_thermal_emergence = 0;       ///< 0 = use ldelay (calendar days), 1 = gate emergence on plant TT
    double tt_emergence = -1.0;          ///< thermal-time threshold for this stem to emerge [deg Cd], <0 disables

    /* Thermal-time cessation gate (mainstem end-of-elongation at VT).
     * When use_thermal_cessation=1, nodal growth is frozen once plant accumulated TT
     * reaches tt_cessation [degCd]. Implemented by setting delayNGStart=age,
     * delayNGEnd=1e9 at the step where the threshold is crossed — reuses the existing
     * delayNG machinery. One-shot: the latch is never released. tt_cessation<0 disables. */
    int use_thermal_cessation = 0;       ///< 0 = no cessation, 1 = freeze at plant TT >= tt_cessation
    double tt_cessation = -1.0;          ///< thermal-time threshold for elongation to stop [deg Cd], <0 disables

    /* Fournier-Andrieu per-phytomer internode kinetics (opt-in).
     * See PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §B.1.
     * Scope: maize_calibrated.xml mainstem subType=1 only; all other species,
     * test XMLs, and maize tassel (subType=20/21) keep flag=false and use the
     * existing scalar delayNG + cessation_age_ path bit-for-bit (Hard Invariant #5). */
    int use_fournier_andrieu_kinetics = 0;             ///< OPT-IN: enables FA per-phytomer internode kinetics (int so bindParameter can serialize it; truthy semantics unchanged)
    double Tb_kinetic = 9.8;                           ///< °C, per-paper Tb chosen to match F-A 2000 + Birch 2002
    double r_I = 0.023;                                ///< Phase I exponential rate [1/degCd] (Zhu 2014)
    double phase_I_duration = 309.0;                   ///< Phase I duration [degCd] (FA 2000)
    double phase_II_duration = 25.0;                   ///< Phase II duration [degCd] (FA 2000)
    double phase_IV_duration = 30.0;                   ///< Phase IV duration, operational [degCd] (FA 2000)
    double phase_IV_k = 0.09;                          ///< Phase IV exponential decay rate [1/degCd] (FA 2000)
    std::vector<int> basal_zero_ranks = {1, 2, 3, 4};  ///< ranks with IL_final=0 (Zhu 2014, He 2021)
    std::vector<double> internode_v_n;                 ///< per-rank Phase III rate [cm/degCd] (FA 2000 Fig 12A); empty→disabled
    std::vector<double> internode_D_n;                 ///< per-rank Phase III duration [degCd] (FA 2000 Fig 12B); empty→disabled
    std::vector<double> internode_IL_final;            ///< per-rank Phase IV asymptote [cm] (FA 2000 Fig 13 / MF3D); empty→disabled

    /* S3b.7 — Plastochron-driven rank initiation (plan §E.b).
     * Under use_fournier_andrieu_kinetics=true the scalar branching-zone burst
     * (all laterals fire when length >= p.lb) is retired; ranks initiate one at
     * a time when plant Andrieu-TT crosses n * plastochron_andrieu. This
     * decouples node creation from leaf emergence, breaking the S3b.3
     * chicken-and-egg deadlock and producing distinct collar z-positions
     * along the basal zone (V3 = ≥5 distinct basal-zone nodes, not 5 stacked
     * at one point). FA-off path is unchanged. */
    double plastochron_andrieu = 23.0;                 ///< °Cd per rank on Andrieu Tb=9.8 axis (Fournier 2000 Déa)
    double basal_internode_cm = 0.4;                   ///< fixed internode spacing at rank initiation [cm] (≥0.3 cm plan acceptance floor, <0.5 so HI#2 stays within ±0.5 cm vs scalar baseline when p.ln has one zero-padded stub)

    /*
     * Callback functions for the Stem (set up by the class StemSystem)
     */
    std::shared_ptr<SoilLookUp> f_se = std::make_shared<SoilLookUp>(); ///< scale elongation function
    std::shared_ptr<SoilLookUp> f_sa = std::make_shared<SoilLookUp>(); ///< scale angle function
    std::shared_ptr<SoilLookUp> f_sbp = std::make_shared<SoilLookUp>(); ///< scale branching probability functiongrowth

protected:

    void bindParameters() override; ///<sets up class introspection

};

} // end namespace CStemBox

#endif
