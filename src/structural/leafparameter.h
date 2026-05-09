// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#ifndef LEAFPARAMETER_H_
#define LEAFPARAMETER_H_

#include "mymath.h"
#include "soil.h"
#include "organparameter.h"
#include "growth.h"

#include <memory>
#include <vector>

namespace CPlantBox {

class Organism;
class LeafShape;
class LeafShapeDistribution;

/**
 * Parameters of a single leaf (created by LeafSpecificParameter)
 */
class LeafSpecificParameter : public OrganSpecificParameter
{
public:

	LeafSpecificParameter() :OrganSpecificParameter(-1, 0.) { };
	LeafSpecificParameter(int subType, double lb, double la,
	const std::vector<double>& ln, double r, double a, double theta,
	double rlt, double leafArea, bool laterals, double width_blade, double width_petiole):
		OrganSpecificParameter(subType, a) , lb(lb), la(la), r(r),
		theta(theta), rlt(rlt), areaMax(leafArea), laterals(laterals),
		ln(ln), width_blade(width_blade), width_petiole(width_petiole)  { }; ///< Constructor setting all parameters

	/*
	 * Parameters per leaf
	 */
	double lb = 0.; 		///< Basal zone of leaf (leaf-stem) [cm]
	double la = 0.;			///< Apical zone of leaf vein [cm];
	double r = 0.;			///< Initial growth rate [cm day-1]
	double theta = 0.; 		///< Branching angle between veins [rad]
	double rlt = 0.;		///< Leaf life time [day]
	double areaMax = 0.; 	///< Leaf area [cm2]
	bool laterals = false;  ///< Indicates if lateral leaves exist
	std::vector<double> ln = std::vector<double>(); ///< Inter-lateral distances (if laterals) or mid for radial parametrisation (if there are no laterals) [cm]
	double width_blade = 0.;		///< width of leafe blade (cm) = length - lb zone. define later a width growth rate?
	double width_petiole = 0.;		///< width of leafe petiole (cm) = lb zone. define later a width growth rate?

	/* Per-leaf shape evaluator (S2 of PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1).
	 * Set by Leaf::getEffectiveSurfaceCPs / updateNodesFromSurfaceCPs lazily on
	 * first access to a default-fallback MedianLeafShape built from the LRP's
	 * surface_cps grid. The lazy fallback keeps every existing XML byte-identical
	 * (D.0 6-XML invariance gate G1). At S4 LeafRandomParameter::realize() will
	 * pre-populate this with either MedianLeafShape (no distribution) or
	 * ParametricLeafShape (cultivar distribution loaded). `mutable` so the lazy
	 * cache survives the const-correct param() accessor on Leaf. */
	mutable std::shared_ptr<LeafShape> shape;

	int nob() const { return ln.size() + laterals; } //number of laterals = number of phytomers + 1
	double getK() const; ///< Returns the exact maximal leaf length (including leaf stem) of this realization [cm]
	double leafLength() const { return getK()-lb; }; ///< Returns the exact maximal leaf length (excluding leaf stem) of this realization [cm]

	std::string toString() const override;

};



/**
 * A parameter set describing a leaf type
 */
class LeafRandomParameter : public OrganRandomParameter
{
public:

	enum shapeTypes { shape_cylinder = 0, shape_cuboid = 1, shape_2D = 2}; ///< how is the shape of the leaf defined?, see @orgVolume and @orgVolume2Length

	LeafRandomParameter(std::shared_ptr<Organism> plant); ///< default constructor
	virtual ~LeafRandomParameter() { };

    void createGeometry(); // creates the leaf geometry according to parameters

	std::shared_ptr<OrganRandomParameter> copy(std::shared_ptr<Organism> plant) override;

	std::shared_ptr<OrganSpecificParameter> realize() override; ///< Creates a specific leaf from the leaf parameter set

    double nob() const { if(ln>0){ return std::max((lmax-la-lb)/ln+1, 1.);}else{return 1.;} }  ///< returns the mean maximal number of branching nodes [1]
    double nobs() const; ///< returns the standard deviation of number of branches [1]
    double leafLength() { return lmax-lb; }; // lb represents the leaf base
    double leafMid() { return lmax-la-lb; }; //

	std::string toString(bool verbose = true) const override; ///< writes parameter to a string

    void readXML(tinyxml2::XMLElement* element, bool verbose) override; ///< reads a single sub type organ parameter set
    tinyxml2::XMLElement* writeXML(tinyxml2::XMLDocument& doc, bool comments = true) const override; ///< writes a organ leaf parameter set

	/*
	 * Parameters per leaf type
	 */
	double lb = 0.; 	///< Length of petiole [cm]
	double lbs = 0.;  	///< Standard deviation of petiole length[cm]
	double la = 10.;	///< Length between midrib and apex (half of legnth between base and apex) [cm]
	double las = 0.;	///< Standard deviation [cm]
	double ln = 1.; 	///< Inter-subleaf distance [cm]
	double lns = 0.;  	///< Standard deviation inter-subleaf distance [cm]
	int lnf = 0; 		///< type of inter-branching distance (0 homogeneous, 1 linear inc, 2 linear dec, 3 exp inc, 4 exp dec)
    double lmax = 0.;       ///< Maximal leaf length (inlcuding the petiole) [cm]
    double lmaxs = 0.;      ///< Standard deviation of maximal leaf length [cm]
    double areaMax = 10.; 	///< maximal leaf area (reached when stem length reaches lmax) [cm2]
    double areaMaxs = 0.; 	///< Standard deviation of maximal leaf area [cm2]
    double r = 1.;			///< Initial growth rate [cm day-1]
	double rs = 0.;			///< Standard deviation initial growth rate [cm day-1]
	double rotBeta = 0.6;	///< Radial rotation (roll) (rad)
	double betaDev = 0.2;	///< Deviation of radial rotation (rad)
	double initBeta = 0.2;	///< Initial radial rotation (rad)
	int tropismT = 1;		///< Leaf tropism parameter (Type)
	double tropismN = 1.;	///< Leaf tropism parameter (number of trials)
	double tropismS = 0.2;	///< Leaf tropism parameter (mean value of expected changeg) [1/cm]
	double tropismAge = 0.;	///< Leaf tropism parameter (age when switch tropism)
	double tropismAges = 0.;///< Leaf tropism parameter (age when switch tropism, standard deviation)
	double theta = 1.22;	///< Angle between leafvein and parent leafvein (rad)
	double thetas = 0.; 	///< Standard deviation angle between leafvein and parent leafvein (rad)
	double rlt = 1.e9;		///< Leaf life time (days)
	double rlts = 0.;		///< Standard deviation of leaf life time (days)
	double Width_blade = 0.;		///< width of leafe blade (cm) = length - lb zone. define later a width growth rate?
	double Width_blades = 0.;		///< Standard deviation of leaf blade width (cm)
	double Width_petiole = 0.;		///< width of leafe petiole (cm) = lb zone. define later a width growth rate?
	double Width_petioles = 0.;		///< Standard deviation of leaf petiole width (cm)

	/* Parametric leaf shape distribution (S4 of
	 * PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1).
	 *
	 * shape_distribution_path — filesystem path to a cultivar shape JSON
	 *   produced by `dart/coupling/scripts/fit_parametric_leaf_shape.py`.
	 *   Empty (default) keeps the leaf on the MedianLeafShape fallback that
	 *   the S2 lazy path in Leaf::getEffectiveSurfaceCPs already provides;
	 *   non-empty wires every realize() output to a per-plant
	 *   ParametricLeafShape (shared across all 15 ranks of the same plant
	 *   when shape_variation_scale > 0; intercept[rank] verbatim when scale
	 *   is 0). Read by readXML / written by writeXML — handled manually
	 *   alongside surface_cp because OrganRandomParameter::bindParameter
	 *   only supports int* / double*. */
	std::string shape_distribution_path = "";
	/* shape_variation_scale — D11 opt-in knob for the per-plant deviation
	 * draw. 0.0 (default) reproduces today's per-rank median XML at
	 * OBJ-vertex level (G8 byte-identity gate); >0 activates scale * L @ z
	 * variation. Same trust contract as `cultivar_height_factor=1.0,
	 * dev=0.0` for stem height: wired but inert until the user sets it. */
	double shape_variation_scale = 0.0;
	/* shape_rank_index — which rank of the cultivar distribution this
	 * subType corresponds to. -1 (default) means "infer from subType":
	 * for maize the calibrated convention is subType 2..16 → rank 0..14
	 * (plan §S4 step 3). Set explicitly in XML when the subType numbering
	 * differs (e.g. wheat) or when a subType maps to multiple ranks. */
	int shape_rank_index = -1;
	/* Loaded distribution. Populated lazily on the first realize() call
	 * that sees a non-empty shape_distribution_path; subsequent
	 * realize() calls (across all 15 LeafRandomParameter instances of the
	 * same cultivar) hit the process-wide cache in
	 * LeafShapeDistribution::load and share one instance. */
	mutable std::shared_ptr<LeafShapeDistribution> shape_distribution_;
	int gf = 1;				///< Growth function (1=negative exponential, 2=linear)
	int isPseudostem = 0;				///< do the leaf sheaths make a pseudostem? (0 false, 1 true)
	double collarLength = 0.;			///< Length of rigid collar zone where tropism is disabled [cm]
	double collarLengths = 0.;			///< Standard deviation of collar length [cm]
	double tropismExponent = 1.;		///< Exponent for position-dependent tropism strength (1=uniform, >1=tip-heavy)
	double tropismExponents = 0.;		///< Standard deviation of tropism exponent

	/* curvature spline profile: defines position-dependent curvature along the leaf */
	std::vector<double> leafCurvaturePhi = {};   ///< Normalized positions [0,1] along leaf for curvature profile knots
	std::vector<double> leafCurvatureKappa = {}; ///< Curvature magnitudes at each knot position [1/cm]

	/* out-of-plane curvature spline: curvature perpendicular to the growth plane */
	std::vector<double> leafOOPCurvPhi = {};     ///< Normalized positions [0,1] along leaf for OOP curvature knots
	std::vector<double> leafOOPCurvKappa = {};   ///< OOP curvature magnitudes at each knot position [1/cm]

	/* asymmetry spline: left/right width offset along the leaf */
	std::vector<double> leafAsymmetryPhi = {};   ///< Normalized positions [0,1] along leaf for asymmetry knots
	std::vector<double> leafAsymmetryOffset = {};///< Width offset at each knot position [cm]

	/* edge curl spline: margin deflection angle along the leaf */
	std::vector<double> leafEdgeCurlPhi = {};    ///< Normalized positions [0,1] along leaf for edge curl knots
	std::vector<double> leafEdgeCurlAngle = {};  ///< Deflection angle at each knot position [rad]

	/* cross-section curvature spline: transverse curvature profile (positive=concave V/U) */
	std::vector<double> leafCrossSectionPhi = {};  ///< Normalized positions [0,1] along leaf for cross-section knots
	std::vector<double> leafCrossSectionCurv = {}; ///< Cross-section curvature at each knot position

	/* describes the plant geometry */
	std::vector<double> leafGeometryPhi= {}; //2D shape
	std::vector<double> leafGeometryX= {};//2D shape
	int parametrisationType = 0; // 2D shape type : 0 .. radial, 1..along main axis
	int shapeType = 2;  // Shape of the leaf: 0: cylinder (a = radius), 1: cuboid (a = thickness, Width_blade, Width_petiole), 2: 2D (leafGeometryPhi, leafGeometryX, areaMax)

	/* Native 2D leaf-surface NURBS control-point grid (Phase A).
	 * Flat storage: size = surface_n_u * surface_n_v, index = i_u * surface_n_v + i_v.
	 * Coordinates are LEAF-LOCAL (collar at origin, +z along midrib tangent, +x leaf-local
	 * lateral axis = tangent x UP). Empty vector disables the native-surface path and
	 * callers fall back to the 1D leafGeometry/skeleton representation. */
	std::vector<Vector3d> surface_cps = {};
	int surface_n_u = 11;                  ///< number of CPs along the midrib (u-axis)
	int surface_n_v = 5;                   ///< number of CPs across the width (v-axis)
	int surface_deg_u = 3;                 ///< u-direction NURBS degree
	int surface_deg_v = 2;                 ///< v-direction NURBS degree

	/* Fournier coordination / thermal-time elongation (Step 2B) */
	double sl_ratio = 0.4;              ///< sheath:lamina ratio [-] (maize placeholder)
	int use_thermal_elongation = 0;      ///< 0 = backward compat (calendar days), 1 = thermal time
	double T_base = 8.0;                ///< base temperature [deg C]
	double T_opt = 30.0;                ///< optimal temperature [deg C]
	double T_max = 41.0;                ///< ceiling temperature [deg C]
	double LER_max = 1.5;               ///< max leaf elongation rate [mm/degCd]
	double phyllochron_tt = 57.9;        ///< phyllochron in thermal time [deg Cd]
	int use_thermal_emergence = 0;       ///< 0 = use ldelay (calendar days), 1 = gate emergence on plant TT
	double tt_emergence = -1.0;          ///< thermal-time threshold for this leaf to emerge [deg Cd], <0 disables

	/* Leaf-side Fournier-Andrieu logistic length kinetics (PLAN_YOUNG_LEAF_PHYSICS_2026-04-25 §Gap 1)
	 * When use_fa_kinetics=1, leaf length follows
	 *     length(t) = lmax / (1 + exp(-(TT - tau)/sigma))
	 * driven by the plant's accumulated TT (Tb=8 axis, same accumulator as tt_emergence).
	 * Saturates to lmax at large TT → mature renders are bit-identical to the scalar path.
	 * Default off so non-maize XMLs and FA-off regression captures stay untouched.
	 *
	 * S2 / ADR_LEAF_KINEMATICS_2026-04-28 — DEPRECATED. The logistic shadow
	 * branch in Leaf::simulate is retired in S2.C; new calibrations should
	 * use MultiPhaseLeafGrowth (gf=6) with the per-rank Andrieu primitives
	 * declared below. The fields stay populated for one cycle so external
	 * callers (Pheno4D scripts, regression captures) can migrate. */
	int use_fa_kinetics = 0;             ///< 0 = scalar r*dt elongation, 1 = TT-driven logistic
	double tau_extension_n = -1.0;       ///< TT half-max for length kinetics [degCd], <0 disables (acts as "off" sentinel)
	double sigma_extension_n = 60.0;     ///< TT spread (logistic scale) [degCd]

	/* Andrieu, Hillier & Birch (2006) cv. Déa piecewise leaf elongation
	 * kinetics. Per-rank scalars (each leaf subType is one Déa rank — see
	 * ADR_LEAF_KINEMATICS_2026-04-28 §C4 for the position↔subType↔rank
	 * mapping). Active only when this LRP's `gf` field is set to
	 * gft_multi_phase_leaf (=6). The `_n` suffix echoes the JSON column
	 * name (`R1_n`, etc.) and reads as "this rank's R1"; it is NOT a
	 * vector. Default sentinels (0.0 / -1.0) keep MultiPhaseLeafGrowth
	 * inert for any non-Andrieu leaf even if the GF is mistakenly minted.
	 *
	 * Length law (Andrieu et al. 2006 eq. C.3):
	 *   Phase E (exp):   t ∈ [T0_n, T1_n)         → L = L_min · exp(R1_n · (t − T0_n))
	 *   Phase L (lin):   t ∈ [T1_n, T2_n)         → L = L1_n + R2_n · (t − T1_n)
	 *   Plateau:         t ≥ T2_n                 → L = L_fin_n
	 * with T1_n = T0_n + lag_exp_n, T2_n = T1_n + D_lin_n,
	 *      L1_n = L_min · exp(R1_n · lag_exp_n),
	 *      L_fin_n = L1_n + R2_n · D_lin_n.
	 * t is the plant's Andrieu-axis TT (Tb=9.8 °C, see Plant::andrieu_tt_).
	 * L_fin_n is anchored to MF3D `lmax_n` via R2_n rescale at bake time;
	 * runtime reads `lmax = (coordinated_lmax > 0) ? coordinated_lmax :
	 * param()->getK()` so canopy-coordination machinery (M9 / Lock #5)
	 * stays live. */
	double R1_n = 0.0;          ///< Phase E (exponential) rate constant [1/°Cd]; <=0 disables MultiPhaseLeafGrowth
	double R2_n = 0.0;          ///< Phase L (linear) rate [cm/°Cd]; rescaled to honour MF3D L_fin
	double lag_exp_n = 0.0;     ///< Phase E duration on Andrieu axis [°Cd]
	double D_lin_n = 0.0;       ///< Phase L duration on Andrieu axis [°Cd]
	double T0_n = 0.0;          ///< Phase E origin on Andrieu axis [°Cd] = (rank − 1) · plastochron
	double L_min = 0.025;       ///< Phase E initial length at T0_n [cm]; Andrieu et al. 2006 p. 1007
	double t_col_emp_Cd = -1.0; ///< Empirical collar emergence on Andrieu (Tb=9.8) axis [°Cd]; <0 disables (uses computed fallback T0+lag_exp+α·D_lin)

	/* call back functions */
    std::shared_ptr<SoilLookUp> f_se = std::make_shared<SoilLookUp>(); ///< scale elongation function
    std::shared_ptr<SoilLookUp> f_sa = std::make_shared<SoilLookUp>(); ///< scale angle function
    std::shared_ptr<SoilLookUp> f_sbp = std::make_shared<SoilLookUp>(); ///< scale branching probability function

    std::vector<std::vector<double>> leafGeometry; // normalized x - coordinates per along the normalized mid vein
	int geometryN = 100; // leaf geometry resolution (not in XML)

    void createLeafGeometry(std::vector<double> y, std::vector<double> l, int N); // create normalized leaf geometry
    void createLeafRadialGeometry(std::vector<double> phi, std::vector<double> l, int N); // create normalized leaf geometry from a radial parameterization

protected:

    void bindParameters() override; ///<sets up class introspectionbindParameters
    std::vector<double> intersections(double y, std::vector<double> phi, std::vector<double> l); ///< returns the intersection of a horizontal line at y-coordinate with the leaf geometry
    void normalizeLeafNodes(); ///< scales leaf area to 1

};

} // end namespace CPlantBox

#endif
