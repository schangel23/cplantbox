#ifndef LEAF_H_
#define LEAF_H_

#include "Organ.h"
#include "Organism.h"
#include "leafparameter.h"

#include <iostream>
#include <assert.h>

namespace CPlantBox {

class Plant;

/**
 * Leaf
 *
 * Describes a single leaf, by a vector of nodes representing the leaf
 * The method simulate() creates new nodes of this leaf, and lateral leafs in the leaf's branching zone.
 *
 */
class Leaf : public Organ
{
public:

    Leaf(int id,  std::shared_ptr<const OrganSpecificParameter> param, bool alive, bool active, double age, double length,
        Vector3d partialIHeading_, int pni, bool moved = true, int oldNON = 0);
	Leaf(std::shared_ptr<Organism> plant, int type, double delay, std::shared_ptr<Organ> parent, int pni); ///< used within simulation
	virtual ~Leaf() { };

	std::shared_ptr<Organ> copy(std::shared_ptr<Organism> plant) override;   ///< deep copies the root tree

	int organType() const override { return Organism::ot_leaf; } ///< returns the organs type

	void simulate(double dt, bool silence = false) override; ///< leaf growth for a time span of \param dt

    Vector3d getNode(int i) const override { return nodes.at(i); } ///< i-th node of the organ

	double getParameter(std::string name) const override; ///< returns an organ parameter of Plant::ScalarType

	/* leaf vizualisation */
    double leafLength( bool realized = false) const { return std::max(getLength(realized)-param()->lb, 0.); /* represents the leaf base*/ }; ///< leaf surface length [cm]
    double leafCenter( bool realized = false) const { return std::max(getLength(realized)-param()->la-param()->lb, 0.); }; ///< center of the radial parametrisation
    double leafArea( bool realized = false, bool withPetiole = false) const; ///< returns the leaf surface area, zero if there are lateral-leafs [cm2]
	double leafAreaAtSeg(int localSegId, bool realized = false, bool withPetiole = false); //leaf area at a specific segment
	double leafVolAtSeg(int localSegId, bool realized = false, bool withPetiole = false); //leaf area at a specific segment
	double leafLengthAtSeg(int localSegId, bool withPetiole = false);
	std::vector<double> getLeafVisX(int i);
	std::vector<Vector3d> getLeafVis(int i); // per node

    std::string toString() const override;

	/* exact from analytical equations */
	double calcCreationTime(double lenght); ///< analytical creation (=emergence) time of a node at a length
	double calcLength(double age); ///< analytical length of the leaf
	double calcAge(double length) const; ///< analytical age of the leaf

	/* abbreviations */
	std::shared_ptr<LeafRandomParameter> getLeafRandomParameter() const;  ///< root type parameter of this root
	std::shared_ptr<const LeafSpecificParameter> param() const; ///< root parameter

	/* Fournier coordination state (Step 2B) */
	bool emerged = false;            ///< tip exceeded pseudostem height
	bool ligulated = false;          ///< lamina reached lamina_Lmax
	double emergence_tt = -1.0;      ///< thermal time at emergence [deg Cd]
	double ligulation_tt = -1.0;     ///< thermal time at ligulation [deg Cd]
	double accumulated_tt = 0.0;     ///< accumulated thermal time [deg Cd]
	double coordinated_lmax = -1.0;  ///< Lmax from coordination (-1 = use XML default)
	bool lmax_set = false;           ///< Lmax has been set by coordination

	/* Fournier-Andrieu plumbing (plan §B.4). Plant's Andrieu-axis (Tb=9.8)
	 * thermal time sampled at the first simulate() step where this leaf's
	 * age becomes positive (= leaf emergence = internode-n primordium init).
	 * Used by Stem::calcLengthPerPhytomerSum to anchor per-rank FA kinetics.
	 * -1.0 = not emerged yet. Always set when a leaf emerges, regardless of
	 * whether FA kinetics are enabled (cheap bookkeeping, no scalar impact). */
	double emergence_andrieu_tt_ = -1.0;
	double getEmergenceAndrieuTT() const { return emergence_andrieu_tt_; }

	bool hasEmerged() const { return emerged; }
	bool hasLigulated() const { return ligulated; }
	double computePseudostemHeight() const;

	/* useful */
    bool hasMoved() const override { return true; }; ///< always need to update the coordinates of the nodes for the MappedPlant
	double orgVolume(double length_ = -1.,  bool realized = false) const override;
	double orgVolume2Length(double volume_) override;
	bool nodeLeafVis(double l); ///<  leaf base (false), branched leaf (false), or leaf surface area (true)

	/* Native 2D surface CP driver (Phase D). When the LRP carries a
	 * populated surface_cps grid, after createSegments has grown the midrib
	 * we re-project each internal node onto the midrib derived from the
	 * v=0.5 u-line of the length-scaled library CPs. Segment topology is
	 * preserved; only node positions change. Node 0 (collar) stays fixed. */
	bool hasSurfaceCPs() const;
	void updateNodesFromSurfaceCPs();

	/* Young-stage shape blend. Returns the LRP's mature CP grid blended
	 * toward a flat template based on this leaf's maturity (length / lmax).
	 *   alpha = max(0, 1 - (m / kYoungFadeEnd)^kYoungExp),   m in [0, 1]
	 *   cps_eff = (1 - alpha) * mature_cps + alpha * flat_template
	 * The flat template is a zero-y version of mature_cps with midrib z
	 * stretched to the mature arc-length (removes droop without losing
	 * blade length). Mature leaves (alpha≈0) return lrp->surface_cps
	 * bit-for-bit via early-out. */
	std::vector<Vector3d> getEffectiveSurfaceCPs() const;

	/* Tunable knobs for the maturity blend. Set above the mature
	 * threshold you want untouched. kYoungFadeEnd < 1 guarantees no
	 * effect on leaves past that maturity. */
	static constexpr double kYoungFadeEnd = 0.7;
	static constexpr double kYoungExp = 2.0;

protected:

	Vector3d getIncrement(const Vector3d& p, double sdx, int n = -1) override; ///< called by createSegments, to determine growth direction
    Vector3d heading(int n)  const override; ///< current (absolute) heading of the organs at node n
    int getleafphytomerID(int subtype);
    void minusPhytomerId(int subtype);
    void addleafphytomerID(int subtype);
	double beta;

	std::vector<double> getLeafVisX_(double l);
	bool ageDependentTropism = false;///< do we need to check the leaf's age to see when to update the tropism effect?, @see Leaf::rel2abs
    
};

} // namespace CPlantBox

#endif
