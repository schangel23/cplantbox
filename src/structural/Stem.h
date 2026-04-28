#ifndef STEM_H_
#define STEM_H_

#include "Organ.h"
#include "Organism.h"
#include "growth.h"
#include "stemparameter.h"

#include <iostream>
#include <assert.h>

namespace CPlantBox {

class Plant;

/**
 * Stem
 *
 * Describes a single stem, by a vector of nodes representing the stem.
 * The method simulate() creates new nodes of this stem, and lateral stems in the stem's branching zone.
 *
 */
class Stem : public Organ
{
public:


    Stem(int id,  std::shared_ptr<const OrganSpecificParameter> param, bool alive, bool active, double age, double length,
    		Vector3d partialIHeading_, int pni, bool moved = true, int oldNON = 0);
    Stem(std::shared_ptr<Organism> plant, int type, double delay, std::shared_ptr<Organ> parent, int pni); ///< used within simulation
    virtual ~Stem() { };

    std::shared_ptr<Organ> copy(std::shared_ptr<Organism> plant) override;   ///< deep copies the root tree

    int organType() const override { return Organism::ot_stem; } ///< returns the organs type

    void simulate(double dt, bool silence = false) override; ///< stem growth for a time span of \param dt
	void internodalGrowth(double dl,double dt, bool silence = false); ///< internodal growth of \param dl [cm]
	double getLatInitialGrowth(double dt) override;
	double getLatGrowthDelay(int ot_lat, int st_lat, double dt) const override;
	Vector3d getNode(int i) const override { return nodes.at(i); } ///< i-th node of the organ
	void addNode(Vector3d n, int id, double t, size_t index, bool shift) override; //< adds a node to the root

    double getParameter(std::string name) const override; ///< returns an organ parameter
	std::string toString() const override;

	/* Phase E: parent-stem radius lookup used by the leaf sheath CP grid.
	 * Current stems carry a single radius parameter ``a`` (no native taper),
	 * so ``arc_length`` is accepted for API symmetry and forward compatibility
	 * but not consulted. Returns cm. */
	double getRadiusAt(double arc_length) const;

    /* exact from analytical equations */
    double calcLength(double age); ///< analytical length of the stem
    double calcAge(double length) const; ///< analytical age of the stem

    /* Fournier-Andrieu per-phytomer internode kinetics (plan §B.3).
     * Only consulted when the stem's StemSpecificParameter has
     * use_fournier_andrieu_kinetics=true (default false — no effect on scalar
     * path). Anchors tau at per-rank internode initiation
     * (leaf-n emergence + 9.6 °Cd half-plastochron lag, FA 2000 line 207). */
    double calcLengthPerPhytomer(int n) const;
    double calcLengthPerPhytomerSum() const;

    /* S3b full per-phytomer bookkeeping (plan §A). Returns insertion index for
     * rank n's next node = "one past the last existing node that belongs to
     * rank n-1". Fallback nodes.size() when rank n-1 has no nodes (cold-start:
     * ranks 1-4 are basal-zero so the first rank to initiate (rank 5) falls
     * through to apex append). Only meaningful when the FA flag is true and
     * node_to_phytomer is populated. */
    int computeInsertionIndexForRank(int n) const;

    /* Per-rank latched realised length [cm] accessor for validation tests
     * (plan §A, used by S3b.5 per-rank τ_n-axis overlay). Returns the latched
     * length of rank n (monotone per Decision 2), 0.0 for basal-zero ranks,
     * 0.0 when the FA flag is false. */
    double getPhytomerLength(int n) const;

    /* S0.5b: GF-side per-organ FA state accessor. Returns a pointer to the
     * MultiPhaseStemGrowth::PerOrganFAState entry for this organ id, or
     * nullptr when the stem is not FA-on (no MultiPhaseStemGrowth GF, or
     * not yet seeded by a getLength() call). After S0.5b state migration
     * the GF entry is the source of truth; the Stem mirror fields are
     * scheduled for retirement. */
    MultiPhaseStemGrowth::PerOrganFAState* getFaState() const;

    /* abbreviations */
    std::shared_ptr<StemRandomParameter> getStemRandomParameter() const;  ///< root type parameter of this root
    std::shared_ptr<const StemSpecificParameter> param() const; ///< root parameter

    std::vector<int> stemphytomerId = std::vector<int>(30, 0);  // indexed by child subType; tassel uses subTypes 20/21, so must cover >=22
    int shootborneType = 5;

																										 
    std::vector<int> getLocalIdLinkingNodes() const { return localId_linking_nodes; } ///< expose linking node local IDs for post-injection sync

    std::shared_ptr<Organ> getChildByPhytomerRank(int rank, int organType, bool isSheath) const; ///< find child by phytomer rank + sheath/blade parity

    /// Age [day] at which thermal-time cessation latched for this stem.
    /// -1 = not triggered yet. Set in simulate() when plant TT crosses
    /// tt_cessation; read in the length-increment block to clamp age__
    /// (freezes nodal elongation, same semantics as delayNGStart).
    double cessation_age_ = -1.0;

    /// Plant's Andrieu-axis TT (Tb=9.8) at the step the cessation gate latched.
    /// -1 = not triggered. Used by the FA branch of simulate() to freeze
    /// per-rank kinetics analogously to the scalar path's cessation_age_ clamp.
    /// Always sampled when cessation_age_ fires; harmless for the scalar path.
    double cessation_andrieu_tt_ = -1.0;

    /* S3b per-phytomer bookkeeping (plan §A, decisions 1–5).
     *
     * All six vectors are EMPTY when use_fournier_andrieu_kinetics is false,
     * so scalar-path stems pay zero memory or runtime overhead. Lazy-sized on
     * the first FA-on simulate() step to n_ranks+1 entries (index 0 unused,
     * ranks indexed 1..n_ranks).
     *
     * Hard Invariant #5 (plan): getLength() == basal_length_ + Σ length_per_n
     * after every simulate(dt) step. basal_length_ accounts for the p.lb
     * bootstrap stub growth driven by the scalar basal-zone allocator (kept
     * under Option 1 bootstrap, see Stem::simulate FA branch). */
    std::vector<double> length_per_n;              ///< realised latched length of each internode [cm]
    std::vector<double> epsilonDx_per_n;           ///< per-phytomer sub-resolution remainder [cm]
    std::vector<double> cessation_age_per_n;       ///< per-rank cessation latch on legacy Tb=8 axis [day]
    std::vector<double> cessation_andrieu_tt_per_n;///< per-rank cessation latch on Andrieu Tb=9.8 axis [degCd]
    std::vector<int>    node_to_phytomer;          ///< parallel to nodes; rank (>=1) owning each node, 0 = basal stub / untagged
    std::vector<char>   lateral_spawned_per_n;     ///< bool flag per rank, set when rank n's lateral has been spawned (char avoids std::vector<bool> bit-packing)

    /// S3b.7 — Plant-Andrieu-TT at the step rank n was initiated via the
    /// plastochron clock. Used as the τ_n anchor for calcLengthPerPhytomer(n)
    /// when the rank was created by the plastochron-driven initiation path,
    /// decoupling internode kinetics from leaf emergence (resolves the
    /// S3b.3 chicken-and-egg deadlock). Entry <0 means "not yet initiated
    /// on the plastochron clock"; calcLengthPerPhytomer then falls back to
    /// the leaf-emergence lookup (legacy S3b.3 path, kept for back-compat).
    std::vector<double> initiation_andrieu_tt_per_n;

    /// Growth accumulated in the p.lb basal-stub zone [cm]. Under Option 1
    /// bootstrap the scalar basal-zone allocator grows the stem from 0 → p.lb
    /// before any FA rank contributes length; those nodes carry rank 0 in
    /// node_to_phytomer and their summed length is tracked here so that the
    /// total-length invariant closes: getLength() == basal_length_ + Σ length_per_n.
    double basal_length_ = 0.0;

protected:
	void storeLinkingNodeLocalId(int numCreatedLN, bool silence) override; ///<  override by @see Organ::createNonGrowingLateral()
	std::vector<int> localId_linking_nodes;
	void minusPhytomerId(int subtype) { stemphytomerId[subtype]--;  }
    int getphytomerId(int subtype) { return stemphytomerId[subtype]; }
    void addPhytomerId(int subtype) { stemphytomerId[subtype]++;  }

};

} // namespace CPlantBox

#endif
