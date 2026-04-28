#ifndef GROWTH_H
#define GROWTH_H

#include <memory>
#include <vector>
#include "Organ.h"
#include "Organism.h"

namespace CPlantBox {

// Forward declarations for MultiPhaseStemGrowth (defined further below).
// Bodies live in growth.cpp where Stem.h / Leaf.h / Plant.h can be included
// without creating a circular include with structural/Stem.h.
class Stem;
class StemRandomParameter;
class StemSpecificParameter;

/**
 * Abstract base class to all growth functions: currently LinearGrowth and ExponentialGrowth
 */
class GrowthFunction
{
public:
	virtual ~GrowthFunction() {};

	std::map<int, double> CW_Gr;

	/**
	 * Returns root length at root age t
	 *
	 * @param t     organ age [day]
	 * @param r     initial growth rate [cm/day]
	 * @param k     maximal organ length [cm]
	 * @param root  points to the organ in case more information is needed
	 *
	 * \return      organ length [cm] at specific age
	 */
	virtual double getLength(double t, double r, double k, std::shared_ptr<Organ> o) const
	{ throw std::runtime_error( "getLength() not implemented" ); return 0; } ///< Returns root length at root age t

	/**
	 * Returns the age of a root of length l
	 *
	 * @param l     organ length [cm]
	 * @param r     initial growth rate [cm/day]
	 * @param k     maximal root length [cm]
	 * @param root  points to the organ in case more information is needed
	 *
	 * \return      organ age [day] at specific length
	 */
	virtual double getAge(double l, double r, double k, std::shared_ptr<const Organ> o) const
	{ throw std::runtime_error( "getAge() not implemented" ); return 0; } ///< Returns the age of a root of length l


	virtual std::shared_ptr<GrowthFunction> copy() const { return std::make_shared<GrowthFunction>(*this); } ///< Copy the object
};



/**
 * LinearGrowth elongates at constant rate until the maximal length k is reached
 */
class LinearGrowth : public GrowthFunction
{
public:

	double getLength(double t, double r, double k, std::shared_ptr<Organ> o) const override { return std::min(k,r*t); } ///< @copydoc GrowthFunction::getLegngth

	double getAge(double l, double r, double k, std::shared_ptr<const Organ> o)  const override { return l/r; } ///< @copydoc GrowthFunction::getAge

	std::shared_ptr<GrowthFunction> copy() const override { return std::make_shared<LinearGrowth>(*this); } ///< @copydoc GrowthFunction::copy

};


/**
 * ExponentialGrowth elongates initially at constant rate r and slows down towards the maximum length k
 */
class ExponentialGrowth : public GrowthFunction
{
public:

	double getLength(double t, double r, double k, std::shared_ptr<Organ> o) const override { return k*(1-exp(-(r/k)*t)); } ///< @copydoc GrowthFunction::getLegngth

	double getAge(double l, double r, double k, std::shared_ptr<const Organ> o) const override { ///< @copydoc GrowthFunction::getAge
		double age = - k/r*log(1-l/k);
		if (std::isfinite(age)) { // the age can not be computed when root length approaches max length
		    return age;
		} else {
		    return 1.e9; // very old
		}
	} ///< @see GrowthFunction

	std::shared_ptr<GrowthFunction> copy() const override { return std::make_shared<ExponentialGrowth>(*this); }

};


/**
 * CWLimitedGrowth uses growth given by phloem module
 */
class CWLimitedGrowth : public ExponentialGrowth
{
public:

	double getLength(double t, double r, double k, std::shared_ptr<Organ> o) const override {
		double length_;
		if (this->CW_Gr.empty()  ){//
			double length = ExponentialGrowth::getLength(t, r, k, o);
			return length;
		} else {
			if((CW_Gr.count(o->getId()) ==0)||(this->CW_Gr.find(o->getId())->second<0)){length_ = 0; //org created at this time step
				if((t> o->getOrganism()->getDt())&&(this->CW_Gr.find(o->getId())->second<-1e-5)){//possible rounding errors?
					assert(false);
				}
			}else{
				length_= o->getLength(false) +this->CW_Gr.find(o->getId())->second; // o->getParameter("length");
				const_cast<double&>( this->CW_Gr.find(o->getId())->second ) = -1.;//sucrose is spent
			}
			return length_;
		}
	} ///< @copydoc GrowthFunction::getLegngth

	double getAge(double l, double r, double k, std::shared_ptr<const Organ> o) const override {
		return ExponentialGrowth::getAge(l, r, k, o);//used to compute growth delay of root and leaf laterals
	}  ///< @copydoc GrowthFunction::getAge

	std::shared_ptr<GrowthFunction> copy() const override { return std::make_shared<CWLimitedGrowth>(*this); }

};

/**
 * GompertzGrowth: sigmoid growth curve with lag phase, inflection, and asymptote.
 * L(t) = K * exp(-exp(-c*(t - t_m)))
 * where t_m = ln(K/r) / c is the inflection time (derived so initial slope ≈ r).
 * Parameter c controls the steepness; derived from r and K as c = r*e/K.
 */
class GompertzGrowth : public GrowthFunction
{
public:

	double getLength(double t, double r, double k, std::shared_ptr<Organ> o) const override {
		double e_ = std::exp(1.0);
		double c = r * e_ / k;           // steepness from initial growth rate
		double t_m = std::log(k / r) / c; // inflection time
		return k * std::exp(-std::exp(-c * (t - t_m)));
	}

	double getAge(double l, double r, double k, std::shared_ptr<const Organ> o) const override {
		if (l <= 0) return 0;
		if (l >= k) return 1.e9;
		double e_ = std::exp(1.0);
		double c = r * e_ / k;
		double t_m = std::log(k / r) / c;
		double age = t_m - std::log(-std::log(l / k)) / c;
		if (std::isfinite(age) && age >= 0) {
			return age;
		} else {
			return 1.e9;
		}
	}

	std::shared_ptr<GrowthFunction> copy() const override { return std::make_shared<GompertzGrowth>(*this); }

};


/**
 * MultiPhaseStemGrowth — per-rank multi-phase stem elongation
 * (Fournier & Andrieu 2000 Phase I→IV decomposition for cv. Déa).
 * Dispatches through the native f_gf->getLength chain instead of the
 * legacy `if (use_fournier_andrieu_kinetics)` shadow branch in
 * Stem::simulate.
 *
 * Per-organ kinetic state (per-rank latched length, cessation latches,
 * plastochron-init TT, etc.) lives on this GF instance, keyed by
 * organId. Mirrors the precedent set by CWLimitedGrowth.CW_Gr; non-FA
 * stems never instantiate this GF and pay zero memory or runtime cost.
 *
 * Architectural separation:
 *   - getLength returns a pure scalar target length [cm]; idempotent
 *     under repeated calls in the same step.
 *   - Geometry side effects (createSegments, createLateral, the post-hoc
 *     node_to_phytomer span-walk) stay in Stem::simulate after the
 *     dispatch (Lock #4 of ADR_LEAF_KINEMATICS_2026-04-28).
 *   - syncStateFromGeometry() is a public sink: Stem::simulate calls it
 *     after the geometry side effects so length_per_n[n] tracks
 *     realised segment lengths (closes Hard Invariant #5).
 *   - getAge is the closed-form piecewise inverse with a null-safe
 *     early return for Plant::initCallbacks's nullptr probe (Lock #2).
 *
 * Implementation lives in growth.cpp because porting requires Stem.h /
 * Leaf.h / Plant.h, which would create a circular include if pulled
 * into this header.
 */
class MultiPhaseStemGrowth : public GrowthFunction
{
public:

	/**
	 * Per-organ FA bookkeeping. Indexed 1-based by rank; index 0 unused
	 * (matches the existing Stem::length_per_n convention).
	 */
	struct PerOrganFAState
	{
		// Note: the global cessation_age + cessation_andrieu_tt latches are
		// kept on Stem (cessation_age_ / cessation_andrieu_tt_) because the
		// use_thermal_cessation feature is not FA-specific (non-FA stems
		// with use_thermal_cessation=1 also rely on those fields).
		std::vector<double> length_per_n;                ///< per-rank realised latched length [cm] (monotone)
		std::vector<double> epsilonDx_per_n;             ///< per-phytomer sub-resolution remainder [cm]
		std::vector<double> cessation_age_per_n;         ///< per-rank cessation latch on legacy axis [day]
		std::vector<double> cessation_andrieu_tt_per_n;  ///< per-rank cessation latch on Andrieu axis [°Cd]
		std::vector<char>   lateral_spawned_per_n;       ///< per-rank "lateral fired" flag (char avoids vector<bool>)
		std::vector<double> initiation_andrieu_tt_per_n; ///< per-rank plastochron-init TT [°Cd]; <0 = pre-init
		double basal_length = 0.0;                       ///< p.lb basal-stub growth accumulator [cm]
	};

	/// Per-organ FA state. Mutable so it can be updated from the const
	/// getLength signature inherited from GrowthFunction (same trick as
	/// CWLimitedGrowth's CW_Gr mutation via const_cast).
	mutable std::map<int, PerOrganFAState> per_organ_state;

	/**
	 * Returns the analytical target length for an FA-on stem [cm], at
	 * the plant's current state. Idempotent under repeated calls.
	 *
	 * Behaviour:
	 *   - Lazy-allocates per_organ_state[id] on first call (sized to
	 *     n_ranks+1 from the LRP's per-rank arrays).
	 *   - Updates per-rank cessation latches when plant Andrieu-TT
	 *     crosses the per-rank Phase IV threshold (or, for
	 *     tt_cessation>0, the legacy global plant-TT threshold).
	 *   - When all per-rank cessation latches are set, fires the global
	 *     cessation_age latch.
	 *   - Updates plastochron forecast (seeds length_per_n[n] =
	 *     basal_internode_cm for ranks whose plastochron birthday has
	 *     crossed but not yet been spawned).
	 *   - Computes fa_sum = Σ max(calcLengthPerPhytomer(n), length_per_n[n]).
	 *   - Returns p.lb + min(fa_sum, Σ p.ln) (capped at branching-zone
	 *     capacity). The caller (Stem::simulate via calcLength) adds
	 *     epsilonDx afterward.
	 *   - When global cessation_age is latched, returns the organ's
	 *     current realised length (forces dl=0 in the caller).
	 *
	 * Falls back to the ExponentialGrowth formula for organs whose SP
	 * has the FA flag off (defensive — the factory normally only
	 * instantiates this GF for FA-on stems).
	 */
	double getLength(double t, double r, double k, std::shared_ptr<Organ> o) const override;

	/**
	 * Closed-form piecewise inverse of the FA length law. Null-safe
	 * for Plant::initCallbacks's gf->getAge(1,1,1,nullptr) probe.
	 */
	double getAge(double l, double r, double k, std::shared_ptr<const Organ> o) const override;

	std::shared_ptr<GrowthFunction> copy() const override
	{
		return std::make_shared<MultiPhaseStemGrowth>(*this);
	}

	/**
	 * Re-syncs per_organ_state[id].length_per_n from realised segment
	 * geometry. Called by Stem::simulate after createSegments fires, so
	 * the next step's getLength sees the actual allocated lengths
	 * (Hard Invariant #5).
	 *
	 * @param o                 the stem organ (in relative coordinates)
	 * @param node_to_phytomer  Stem's geometry annotation: rank that owns each node
	 * @param basal_length      growth accumulated in the p.lb stub zone [cm]
	 */
	void syncStateFromGeometry(std::shared_ptr<const Organ> o,
	                           const std::vector<int>& node_to_phytomer,
	                           double basal_length) const;

	/**
	 * Per-rank latched length [cm] for the given organ. Mirrors today's
	 * Stem::getPhytomerLength(n). Returns 0 for empty state, basal-zero
	 * ranks, n out of range, or an organ never seen by getLength.
	 */
	double getPhytomerLength(int organId, int n) const;

	/**
	 * Sum of per-rank latched lengths for the given organ. Mirrors
	 * today's Stem::calcLengthPerPhytomerSum(). Geometry-side readers
	 * (Stem::simulate's invariant checks) use this.
	 */
	double calcLengthPerPhytomerSum(int organId, std::shared_ptr<const Organ> o) const;

	/**
	 * Per-rank target length for rank n given the organ's FA calibration.
	 * Mirrors today's Stem::calcLengthPerPhytomer(n). Public so
	 * Stem::simulate's plastochron-driven createLateral loop can query
	 * "what length would rank n have right now if it had just initiated".
	 */
	double calcLengthPerPhytomer(int n, std::shared_ptr<const Organ> o) const;

private:

	/// Lazy-allocate per_organ_state[organId] sized from the organ's LRP.
	/// Idempotent. Returns the entry by reference.
	PerOrganFAState& ensureState(int organId, std::shared_ptr<const Organ> o) const;
};


} // end namespace CPlantBox


#endif
