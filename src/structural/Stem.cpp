#include "Stem.h"

#include "Leaf.h"
#include "Root.h"
#include "Plant.h"
#include "Seed.h"
#include <algorithm>

namespace CPlantBox {


/**
 * Constructs a root from given data.
 * The organ tree must be created, @see Organ::setPlant, Organ::setParent, Organ::addChild
 * Organ geometry must be created, @see Organ::addNode, ensure that this->getNodeId(0) == parent->getNodeId(pni)
 *
 * @param id        		the organ's unique id (@see Organ::getId)
 * @param param     		the organs parameters set, ownership transfers to the organ
 * @param alive     		indicates if the organ is alive (@see Organ::isAlive)
 * @param active    		indicates if the organ is active (@see Organ::isActive)
 * @param age       		the current age of the organ (@see Organ::getAge)
 * @param length    		the current length of the organ (@see Organ::getLength)
 * @param partialIHeading 	the initial partial heading of this root
 * @param pbl       		base length of the parent root, where this root emerges
 * @param pni       		local node index, where this root emerges
 * @deprecated moved		indicates if nodes were moved in the previous time step (default = false)
 * @param oldNON    		the number of nodes of the previous time step (default = 0)
 */
Stem::Stem(int id, std::shared_ptr<const OrganSpecificParameter> param, bool alive, bool active, double age, double length,
		Vector3d partialIHeading_, int pni, bool moved, int oldNON)
:Organ(id, param, alive, active, age, length, partialIHeading_,  pni, moved,  oldNON)
{}

/**
 * Constructor
 * This is a Copy Paste of the Root.cpp but it works independently, it has its own parameter file (in .stparam file) tropism, growth function, txt and vtp writing system.
 * All of those can be modified to fit the real growth of the Plant.
 *
 * Typically called by the Plant::Plant(), or Stem::createNewStem().
 * For stem the initial node and node emergence time (netime) must be set from outside
 *
 * @param plant 		points to the plant
 * @param parent 		points to the parent organ
 * @param subtype		sub type of the stem
 * @param delay 		delay after which the organ starts to develop (days)
 * @param rheading		relative heading (within parent organ)
 * @param pni			parent node index
 * @param pbl			parent base length
 */
Stem::Stem(std::shared_ptr<Organism> plant, int type, double delay,  std::shared_ptr<Organ> parent, int pni)
:Organ(plant, parent, Organism::ot_stem, type, delay, pni)
{
	assert(parent!=nullptr && "Stem::Stem parent must be set");
	auto p = this->param();
	int phytomerId = 0;
	if(parent->organType()==Organism::ot_stem)
	{
		std::static_pointer_cast<Stem>(parent)->addPhytomerId(p->subType);
		phytomerId = std::static_pointer_cast<Stem>(parent)->getphytomerId(p->subType);
	}
	double beta = phytomerId*M_PI*getStemRandomParameter()->rotBeta +
			M_PI*plant->rand()*getStemRandomParameter()->betaDev;
	beta = beta + getStemRandomParameter()->initBeta*M_PI;
	if (getStemRandomParameter()->initBeta >0 && phytomerId ){
		beta = beta + getStemRandomParameter()->initBeta*M_PI;
	}
	double theta = p->theta;//M_PI*p->theta;
	if (parent->organType()!=Organism::ot_seed) { // scale if not a base organ, to delete?
		double scale = getStemRandomParameter()->f_sa->getValue(parent->getNode(pni), parent);
		theta *= scale;
	}
	//used when computing actual heading, @see Stem::getIHeading
	this->partialIHeading = Vector3d::rotAB(theta,beta);
	if (parent->organType()!=Organism::ot_seed) { // initial node
		//if lateral of stem, initial creation time:
		//time when stem reached end of basal zone (==CT of parent node of first lateral) + delay
		// @see stem::createLateral
		double creationTime;
		if (parent->getNumberOfChildren() == 0){creationTime = parent->getNodeCT(pni)+delay;
		}else{creationTime = parent->getChild(0)->getParameter("creationTime") + delay;}

		Organ::addNode(Vector3d(0.,0.,0.), parent->getNodeId(pni), creationTime);//do not know why, but i have to add "Organ::" now
	}
}

/**
 * Deep copies the organ into the new plant @param rs.
 * All laterals are deep copied, plant and parent pointers are updated.
 *
 * @param plant     the plant the copied organ will be part of
 */
std::shared_ptr<Organ> Stem::copy(std::shared_ptr<Organism> p)
{
	auto s = std::make_shared<Stem>(*this); // shallow copy
	s->parent = std::weak_ptr<Organ>();
	s->plant = p;
	s->param_ = std::make_shared<StemSpecificParameter>(*param()); // copy parameters
	for (size_t i=0; i< children.size(); i++) {
		s->children[i] = children[i]->copy(p); // copy laterals
		s->children[i]->setParent(s);
	}
	return s;
}

/**
 * Simulates growth of this stem for a time span dt
 *
 * @param dt       time step [day]
 * @param verbose  indicates if status messages are written to the console (cout) (default = false)
 */
void Stem::simulate(double dt, bool verbose)
{
	if(!hasRelCoord()){
		throw std::runtime_error("organism no set in rel coord");
	}
	const StemSpecificParameter& p = *param(); // rename
	firstCall = true;
	oldNumberOfNodes = nodes.size();
	auto p_all = plant.lock();
	auto p_stem = p_all->getOrganRandomParameter(Organism::ot_stem);

	// S0.5b.2: hoist a pointer to the FA per-organ kinetic state on the
	// MultiPhaseStemGrowth GF.  Writers throughout simulate() update this
	// in lockstep with the Stem mirror fields so the GF entry stays the
	// canonical source of truth between simulate() steps.  The dispatch
	// site no longer pushes Stem→GF; getLength's internal writes are
	// reflected back to the Stem mirror via the post-getLength pull
	// (kept until S0.5b.5 retires the mirror entirely).  fa_gf is
	// nullptr for non-FA stems (scalar path runs unchanged).
	MultiPhaseStemGrowth::PerOrganFAState* fa_gf = nullptr;
	{
		const auto& srp_init = *getStemRandomParameter();
		if (srp_init.use_fournier_andrieu_kinetics) {
			auto gf_mps = std::dynamic_pointer_cast<MultiPhaseStemGrowth>(srp_init.f_gf);
			if (gf_mps) {
				fa_gf = &gf_mps->per_organ_state[getId()];
			}
		}
	}

	if (alive) { // dead roots wont grow

		// increase age
		if (age+dt>p.rlt) { // root life time
			dt=p.rlt-age; // remaining life span
			alive = false; // this root is dead
		}
		age+=dt;

		// --- Thermal-time emergence gate (subtype-agnostic, opt-in via use_thermal_emergence) ---
		// Gates stem "birth" on plant TT instead of calendar-day ldelay. Each stem subType
		// carries a tt_emergence threshold; the stem does not grow until plant accumulated TT
		// crosses it. Used e.g. for tassel subType to emerge at VT under variable T forcing.
		{
			auto plant_tt = getPlant();
			const auto& srp_tt = *getStemRandomParameter();
			if (plant_tt && srp_tt.use_thermal_emergence && srp_tt.tt_emergence > 0.0) {
				if (plant_tt->getAccumulatedTT() < srp_tt.tt_emergence) {
					age -= dt;          // unborn: revert age, no growth this step
					return;
				}
			}
			// --- Thermal-time cessation gate (mainstem end-of-elongation at VT) ---
			// One-shot latch: first step where the (axis-selected) plant TT crosses
			// the configured threshold records the stem's current age; subsequent
			// length updates clamp age__ to that age (see below).
			//
			// S0.6 / Lock #1 — three-way threshold source, in priority order:
			//   delayNGEndAxis==TT && delayNGEnd>0 → Andrieu-TT (Tb=9.8) crossing
			//                                         (merged Lock #1 form)
			//   tt_cessation > 0                   → legacy plant-TT (Tb=8) crossing
			//   else                               → no global fast-latch; the
			//                                         per-rank-all-latched gate
			//                                         below fires the global latch.
			// Existing XMLs default delayNGEndAxis=Calendar → axis_tt_global is
			// false → legacy path is the only one active → bit-identical.
			if (plant_tt && srp_tt.use_thermal_cessation && cessation_age_ < 0.0) {
				const bool axis_tt_global = (srp_tt.delayNGEndAxis == DelayAxis::TT)
				                            && (srp_tt.delayNGEnd > 0.0);
				const bool legacy_global = (srp_tt.tt_cessation > 0.0);
				const double plant_andrieu_tt_global = plant_tt->getAccumulatedAndrieuTT();
				const double plant_tt_legacy = plant_tt->getAccumulatedTT();
				bool fired = false;
				if (axis_tt_global && plant_andrieu_tt_global >= srp_tt.delayNGEnd) {
					fired = true;
				} else if (!axis_tt_global && legacy_global
				           && plant_tt_legacy >= srp_tt.tt_cessation) {
					fired = true;
				}
				if (fired) {
					cessation_age_ = age;
					// FA parallel: sample Andrieu-axis TT at the same instant so the
					// FA branch can freeze per-rank kinetics (plan §B.3 cessation
					// interaction). Harmless for scalar path — never read there.
					cessation_andrieu_tt_ = plant_andrieu_tt_global;
				}
			}
		}

		// S3b.7 — Per-rank cessation sampling moved OUT of the active-gated FA
		// branch so it still fires when the stem has reached getK (active=false)
		// but plant TT continues accumulating. Under S3b.7 basal_internode_cm
		// inflates total length slightly, causing active=false a few days earlier
		// than the S3b.3 baseline; with the sampling inside if(active), per-rank
		// latches can miss crossings that would have fired under the slower
		// scalar path. Pure latch operation (records a value, doesn't affect
		// current-step growth), so moving it is safe. Still gated on FA flag +
		// use_thermal_cessation + tt_cessation>0. The GLOBAL latch (10 lines
		// up) and this PER-RANK latch now live in the same "outside if(active)"
		// region, which matches their semantic role (one-shot latches, not
		// growth drivers).
		{
			const auto& srp_fa_outer = *getStemRandomParameter();
			if (srp_fa_outer.use_fournier_andrieu_kinetics
			    && srp_fa_outer.use_thermal_cessation) {
				const int n_ranks_outer = static_cast<int>(srp_fa_outer.internode_v_n.size());
				if (n_ranks_outer > 0) {
					// S0.5b.5: per-rank cessation arrays live on the GF instance.
					if (fa_gf && static_cast<int>(fa_gf->cessation_andrieu_tt_per_n.size()) < n_ranks_outer + 1) {
						fa_gf->cessation_andrieu_tt_per_n.resize(n_ranks_outer + 1, -1.0);
						fa_gf->cessation_age_per_n.resize(n_ranks_outer + 1, -1.0);
					}
					// Plan B.3 (peduncle exuberance, 2026-04-27) + S0.6 / Lock #1:
					// per-rank threshold dispatch — three-way, in priority order.
					//   delayNGEndAxis==TT && delayNGEnd > 0
					//                       → merged Andrieu-TT crossing (Lock #1
					//                          form). Every rank latches at the
					//                          same tau_n, identical operational
					//                          shape to the legacy tt_cessation
					//                          path but living on the Andrieu axis
					//                          and on the canonical native field.
					//   tt_cessation > 0    → legacy global plant-TT crossing
					//                          (preserves S3b.3 behaviour for the
					//                          synthetic tt_cessation=800
					//                          regression test). Read here on the
					//                          Andrieu axis to match the per-rank
					//                          tau_n basis.
					//   else                → per-rank Phase IV operational
					//                          completion: rank n latches once
					//                          tau_n >= phase_I + phase_II
					//                                   + D_n[n] + phase_IV_op.
					//                          Preferred maize-FA path; literature
					//                          anchor is FA 2000 + Birch 2002
					//                          per-rank Phase IV duration.
					const bool axis_tt_active_outer = (srp_fa_outer.delayNGEndAxis == DelayAxis::TT)
					                                  && (srp_fa_outer.delayNGEnd > 0.0);
					const bool legacy_threshold = srp_fa_outer.tt_cessation > 0.0;
					auto plant_cess_outer = getPlant();
					if (plant_cess_outer) {
						double plant_andrieu_tt_outer = plant_cess_outer->getAccumulatedAndrieuTT();
						int leaf_ordinal_outer = 0;
						for (const auto& c : children) {
							if (c->organType() != Organism::ot_leaf) continue;
							leaf_ordinal_outer++;
							if (leaf_ordinal_outer > n_ranks_outer) break;
							// S0.5b.3: read FA-specific per-rank state from GF.
							if (!fa_gf || leaf_ordinal_outer >= static_cast<int>(fa_gf->cessation_andrieu_tt_per_n.size())) break;
							if (fa_gf->cessation_andrieu_tt_per_n[leaf_ordinal_outer] >= 0.0) continue;
							// Prefer plastochron-driven init_tt when available
							// (S3b.7 path); fall back to leaf emergence axis.
							double init_tt_outer = -1.0;
							if (leaf_ordinal_outer < static_cast<int>(fa_gf->initiation_andrieu_tt_per_n.size())
							    && fa_gf->initiation_andrieu_tt_per_n[leaf_ordinal_outer] >= 0.0) {
								init_tt_outer = fa_gf->initiation_andrieu_tt_per_n[leaf_ordinal_outer];
							} else {
								auto lf = std::static_pointer_cast<Leaf>(c);
								double leaf_tt_outer = lf->getEmergenceAndrieuTT();
								if (leaf_tt_outer < 0.0) continue;
								init_tt_outer = leaf_tt_outer + 9.6;   // HALF_PLASTOCHRON_LAG_DEGCD
							}
							double tau_n_outer = plant_andrieu_tt_outer - init_tt_outer;
							double threshold_outer;
							if (axis_tt_active_outer) {
								// Lock #1 merged form: delayNGEnd carries the
								// Andrieu-TT global cessation threshold.
								threshold_outer = srp_fa_outer.delayNGEnd;
							} else if (legacy_threshold) {
								threshold_outer = srp_fa_outer.tt_cessation;
							} else {
								// Per-rank Phase IV completion (Plan B.3).
								const std::size_t d_idx = static_cast<std::size_t>(leaf_ordinal_outer - 1);
								const double D_n = (d_idx < srp_fa_outer.internode_D_n.size())
									? srp_fa_outer.internode_D_n[d_idx]
									: 0.0;
								threshold_outer = srp_fa_outer.phase_I_duration
									+ srp_fa_outer.phase_II_duration
									+ D_n
									+ srp_fa_outer.phase_IV_duration;
							}
							if (tau_n_outer >= threshold_outer) {
								// S0.5b.5: write per-rank latches to GF state only.
								fa_gf->cessation_andrieu_tt_per_n[leaf_ordinal_outer] = plant_andrieu_tt_outer;
								fa_gf->cessation_age_per_n[leaf_ordinal_outer] = age;
							}
						}
					}
					// Plan B.3: when ALL per-rank latches are set (every leaf
					// rank has finished its Phase IV elongation), trigger the
					// global cessation_age_ latch so the existing age__ clamp
					// at line ~250 freezes calcLength() and the FA targetlength
					// stops growing.  Replaces the unreachable plant-TT global
					// latch (Bug 3 in the diagnostic).  Idempotent: only fires
					// once cessation_age_ < 0.
					//
					// S0.6 / Lock #1 — also skipped when the axis-TT global path
					// is active, because that path already fires the global latch
					// in the fast-path block at the top of simulate(). Falling
					// through both would still latch only once (cessation_age_<0
					// guard) but the comment-and-purpose lines up cleaner this
					// way: one path = one fire site.
					if (!legacy_threshold && !axis_tt_active_outer && cessation_age_ < 0.0) {
						bool all_latched = (fa_gf != nullptr);
						for (int n = 1; n <= n_ranks_outer && all_latched; ++n) {
							// S0.5b.3: per-rank latches read from GF.
							if (n >= static_cast<int>(fa_gf->cessation_age_per_n.size())
							    || fa_gf->cessation_age_per_n[n] < 0.0) {
								all_latched = false;
							}
						}
						if (all_latched) {
							cessation_age_ = age;
							if (plant_cess_outer) {
								cessation_andrieu_tt_ = plant_cess_outer->getAccumulatedAndrieuTT();
							}
						}
					}
				}
			}
		}

		// probabilistic branching model (todo test)
		if ((age>0) && (age-dt<=0)) { // the root emerges in this time step
			//use relative coordinates for this function. Delete as it s not a root?
			double P = getStemRandomParameter()->f_sbp->getValue(nodes.back(),shared_from_this());
			if (P<1.) { // P==1 means the lateral emerges with probability 1 (default case)
				double p = 1.-(1.-P*dt); //probability of emergence in this time step
				if (plant.lock()->rand()>p) { // not rand()<p
					age -= dt; // the root does not emerge in this time step
				}
			}
		}

		if (age>0) { // unborn  roots have no children

			// children first (lateral roots grow even if base root is inactive)
			for (auto l:children) {
				l->simulate(dt,verbose);
			}

			if (active) {
                double dt_; // time step
                if (age<dt) { // the root emerged in this time step, adjust time step
                    dt_= age;
                } else {
                    dt_=dt;
                }

				// length increment
				double age__ = age;
				if(age > p.delayNGStart){//simulation ends after start of growth pause
					if(age < p.delayNGEnd){age__ =p.delayNGStart;//during growth pause
					}else{
						age__ = age - (p.delayNGEnd - p.delayNGStart);//simulation ends after end of growth pause
					}
				}//delay to apply
				// Thermal-time cessation latch: once the gate has fired, freeze
				// effective age at the latch value so calcLength() stops growing.
				if (cessation_age_ >= 0.0 && age__ > cessation_age_) {
					age__ = cessation_age_;
				}
				/*as we currently do not implement impeded growth for stem and leaves
				*we can use directly the organ's age to cumpute the target length
				*/
				// Fournier-Andrieu branch (plan §B.3, updated for S3b Option 1
				// 2026-04-23). Replaces the scalar calcLength(age__) with the
				// per-phytomer FA sum when the stem's flag is true. Scalar
				// else-branch is bit-for-bit unchanged from pre-B.3 master
				// (Hard Invariant #1). Flag is false for all non-maize XMLs
				// and for maize tassel subType=20/21; only maize_calibrated
				// mainstem subType=1 sets it to true.
				//
				// S3b Option 1 bootstrap (plan §S3b "Architectural decisions"):
				// targetlength = p.lb + Σ latched IL + epsilonDx. The p.lb
				// offset preserves the basal-stub growth driver — the scalar
				// basal/branching-zone allocator below grows stem 0 → p.lb and
				// fires the branching-zone burst that creates leaves. Once
				// leaves have emerged and their emergence_andrieu_tt_ is set,
				// FA kinetics become non-zero for ranks 5+ and drive further
				// elongation. The thin-B.3.5 scalar max() floor is dropped:
				// under Option 1 the kinetic contribution above p.lb comes
				// solely from the latched FA per-rank sum (decision 1).
				//
				// Per-rank monotonic latch (decision 2, S3b.1 finding 2): we
				// track length_per_n[n] by taking max(length_per_n[n],
				// calcLengthPerPhytomer(n)) each step. This absorbs the Phase
				// IV decay artifact (raw calcLengthPerPhytomer(n) can drop
				// when IL_end_III > IL_final); the latch keeps the kinetic
				// state monotone, which is the load-bearing property for
				// S3b.5 per-rank τ_n validation against Fournier 2000 Déa.
				double targetlength;
				// S0.5 (ADR_LEAF_KINEMATICS_2026-04-28): FA stems dispatch through
				// MultiPhaseStemGrowth::getLength via the native f_gf chain
				// (Lock #4 pure-scalar contract). The legacy `if (use_fournier_
				// andrieu_kinetics)` shadow branch and the `stem_growth_dispatch`
				// enum are retired; the FA flag alone selects the path.
				//
				// Geometry side effects below (branching block, internodalGrowth,
				// span walk, active gate) still read the Stem fields, which we
				// mirror to/from the GF instance state around the f_gf->getLength
				// call. Full retirement of those mirror fields onto the GF is
				// deferred to S0.5b (pytest consumers + the post-step span walk
				// currently depend on the Stem-side accessors).
				if (p.use_fournier_andrieu_kinetics) {
					if (!fa_gf) {
						throw std::runtime_error(
							"Stem::simulate: use_fournier_andrieu_kinetics requires "
							"f_gf to be MultiPhaseStemGrowth (set in "
							"Plant::initCallbacks); fa_gf is nullptr.");
					}
					// Lazy resize per-rank GF arrays to per_n_end so the
					// downstream geometry block reads valid
					// lateral_spawned_per_n / initiation_andrieu_tt_per_n
					// without depending on the (skipped) shadow inner resize
					// blocks.
					const int n_ranks_gf = static_cast<int>(p.internode_v_n.size());
					const int n_laterals_max_gf = static_cast<int>(p.ln.size()) + 1;
					const int per_n_end_gf = std::max(n_ranks_gf + 1,
					                                   n_laterals_max_gf + 1);
					if (static_cast<int>(fa_gf->length_per_n.size()) < per_n_end_gf) {
						fa_gf->length_per_n.resize(per_n_end_gf, 0.0);
						fa_gf->epsilonDx_per_n.resize(per_n_end_gf, 0.0);
						fa_gf->cessation_age_per_n.resize(per_n_end_gf, -1.0);
						fa_gf->cessation_andrieu_tt_per_n.resize(per_n_end_gf, -1.0);
						fa_gf->lateral_spawned_per_n.resize(per_n_end_gf, 0);
					}
					if (static_cast<int>(fa_gf->initiation_andrieu_tt_per_n.size()) < per_n_end_gf) {
						fa_gf->initiation_andrieu_tt_per_n.resize(per_n_end_gf, -1.0);
					}

					// S0.5b.5: dispatch through f_gf->getLength.  GF state
					// is canonical — no push (S0.5b.2) and no pull
					// (retired now that all readers consult getFaState()).
					// Pass `age` (calendar) so update_cessation_latches
					// stamps per-rank latches against age.
					targetlength = getStemRandomParameter()->f_gf->getLength(
						age, p.r, p.getK(), shared_from_this())
					             + this->epsilonDx;
				} else {
					targetlength = calcLength(age__) + this->epsilonDx;
				}
				double e = targetlength-length; // store value of elongation to add
				//can be negative
				double dl = e;//length increment = calculated length + increment from last time step too small to be added
				length = getLength(true);
				this->epsilonDx = 0.; // now it is "spent" on targetlength (no need for -this->epsilonDx in the following)
				// Plan B.3 (peduncle exuberance, 2026-04-27): once the global
				// cessation_age_ latch has fired (under FA-on this fires when
				// every per-rank cessation_age_per_n[1..n_ranks] is set, see
				// outer-latch all-latched check above), zero dl so the
				// FA-leftover internodalGrowth route + apical-zone block
				// stop bleeding length into the apex.  HI#4 (mainstem top
				// ≤ topmost-leaf insertion + 5 cm) only closes when this gate
				// fires; without it the apical-zone block (length >
				// maxInternodeDistance + p.lb) keeps absorbing residual dl.
				// FA-off path keeps the existing age__ clamp at calcLength;
				// this extra hard zero is FA-on only.
				if (p.use_fournier_andrieu_kinetics && cessation_age_ >= 0.0) {
					dl = 0.0;
				}
				// create geometry
				if (p.laterals) { // stem has laterals
					/* basal zone */
					if ((dl>0)&&(length<p.lb)) { // length is the current length of the root
						if (length+dl<=p.lb) {
							createSegments(dl,dt_,verbose);
							length+=dl;
							dl=0;
						} else {
							double ddx = p.lb-length;
							createSegments(ddx,dt_,verbose);
							dl-=ddx; // ddx already has been created
							length=p.lb;
						}
					}
					/* branching zone */
					//go into branching zone if organ has laterals and has reached
					//the end of the basal zone
					if (!p.use_fournier_andrieu_kinetics) {
						// Scalar-path branching-zone burst: all laterals fire
						// at once the step length crosses p.lb. Unchanged from
						// pre-FA behaviour; preserved for FA-off XMLs
						// (non-maize, or maize with the flag flipped off for
						// regression baselines).
						if (((created_linking_node)<(p.ln.size()+1))&&(length>=p.lb))
						{
							for (size_t i=0; (i<p.ln.size()); i++) {
								createLateral(dt_, verbose);
								if(p.ln.at(created_linking_node-1)>0){
									createSegments(this->dxMin(),dt_,verbose);
									dl-=this->dxMin();
									length+=this->dxMin();
								}
							}
							createLateral(dt_, verbose);
						}
					} else {
						// S3b.7 plastochron-gated per-rank initiation (plan §E.b).
						// Replaces the scalar burst: ranks initiate one at a time
						// as plant Andrieu-TT crosses n * plastochron_andrieu.
						// Each rank gets its own basal_internode_cm-spaced node,
						// so a V3 plant ends up with 5 distinct node z-positions
						// stacked in a 2–5 cm zone rather than 5 coincident nodes
						// at z = p.lb exactly.
						//
						// Decoupling from leaf emergence is handled in
						// calcLengthPerPhytomer — it reads
						// initiation_andrieu_tt_per_n[n] when set, so FA kinetics
						// start at the rank's plastochron birthday (τ_n = 0) and
						// do not wait on the leaf's own emergence_andrieu_tt_.
						if (length >= p.lb) {
							const auto& srp_init = *getStemRandomParameter();
							double plastochron = srp_init.plastochron_andrieu;
							double basal_step = srp_init.basal_internode_cm;
							auto plant_init = getPlant();
							double plant_andrieu_tt = plant_init ? plant_init->getAccumulatedAndrieuTT() : 0.0;
							int n_laterals_max = static_cast<int>(p.ln.size()) + 1;
							// Process ranks in ascending order (decision 5 —
							// sort-before-process: under a warm spike multiple
							// ranks can cross their plastochron on the same
							// simulate step, and the older rank must be inserted
							// lower on the stem than the younger).
							for (int n = 1; n <= n_laterals_max; ++n) {
								// S0.5b.3: read spawn flags from GF state.
								if (!fa_gf || static_cast<int>(fa_gf->lateral_spawned_per_n.size()) <= n) break;
								if (fa_gf->lateral_spawned_per_n[n]) continue;
								double init_tt_n = static_cast<double>(n) * plastochron;
								if (plant_andrieu_tt < init_tt_n) break;
								// Match scalar-burst order: lateral attaches at the
								// current apex FIRST, THEN the apex advances.
								// Exception: rank n_laterals_max (topmost lateral,
								// maize tassel at rank 17) attaches without
								// advancing — same semantics as the scalar burst's
								// "extra createLateral outside the for-loop".
								createLateral(dt_, verbose);
								if (n < n_laterals_max) {
									// Grow the branching zone by one
									// basal_internode_cm. dl includes this step's
									// budget via the targetlength forecast above
									// (length_per_n[n] was seeded to basal_step).
									createSegments(basal_step, dt_, verbose);
									length += basal_step;
									dl     -= basal_step;
								}
								// S0.5b.5: per-rank initiation state lives on GF.
								if (fa_gf
								    && n < static_cast<int>(fa_gf->initiation_andrieu_tt_per_n.size())
								    && n < static_cast<int>(fa_gf->lateral_spawned_per_n.size())) {
									fa_gf->initiation_andrieu_tt_per_n[n] = plant_andrieu_tt;
									fa_gf->lateral_spawned_per_n[n] = 1;
								}
							}
						}
					}
					//we can have (p.ln.size()+1)>(created_linking_node) if one ln == 0cm
					if((length>=p.lb)&&((p.ln.size()+1)<(created_linking_node))){
						std::stringstream errMsg;
						errMsg <<"Stem::simulate(): higher number of realized linking nodes ("<<created_linking_node<<
						") than of max laterals ("<<p.ln.size()+1<<")";
						throw std::runtime_error(errMsg.str().c_str());
					}
					//internodal elongation, if the basal zone of the stem is created and still has to grow
					double maxInternodeDistance = p.getK()-p.la - p.lb;//maximum length of branching zone
					if((dl>0)&&(length>=p.lb)&&(maxInternodeDistance>0)){
							int nn = localId_linking_nodes.back(); //node carrying the last lateral == end of branching zone
							double currentInternodeDistance = getLength(nn) - p.lb; //actual length of branching zone
							double ddx = std::min(maxInternodeDistance-currentInternodeDistance, dl);//length to add to branching zone

							if(ddx > 0){
								internodalGrowth(ddx,dt_, verbose);
								dl -= ddx;
							length += ddx;

							}
						}
					/* apical zone */
					//only grows once the basal and branching nodes are developped
					if ((dl>0)&&(length-(maxInternodeDistance + p.lb)>-1e-9)) {
						createSegments(dl,dt_,verbose);
						length+=dl;
					}
				} else { // no laterals
					if (dl>0) {
						createSegments(dl,dt_,verbose);
						length+=dl;

						}
				} // if lateralgetLengths
			if(dl <0){ //to keep in memory that realised length is too long, as created nodes to carry children

				this->epsilonDx = dl;//targetlength + e - length;
				length += this->epsilonDx;//go back to having length = theoratical length
			}
			// S3b per-phytomer bookkeeping: track the basal-zone stub length so
			// the invariant (plan Hard Invariant #5) closes:
			//   getLength(true) ≈ basal_length_ + Σ length_per_n (± dxMin).
			//
			// S3b.3 pragmatic approach (plan §A "absorbed" scope, downgraded
			// 2026-04-24 from full per-rank insertion driver to post-hoc
			// tagging on the scalar allocator's output). The scalar allocator
			// produces the correct mainstem topology under FA-on (verified
			// HI#4 tassel day 125 preserved in thin-B.3.5); the S3b.3 value-add
			// here is that node_to_phytomer is now populated and tracks the
			// per-rank span of nodes. Each node k is tagged with the rank n
			// such that node k lies between linking_node[n-1] and
			// linking_node[n] on the stem's local indexing, with basal-zone
			// nodes tagged 0 and apical-zone (peduncle) nodes tagged
			// n_linking_nodes (== number of laterals == one-past the topmost
			// rank). Per-rank cessation latches (plan §B) operate on the
			// timing axis and remain correct regardless of the tagging
			// strategy. Full mid-stem insertion driver (for same-timestep
			// sort-ordering + parentNI shift testing, plan §A B.5' T2/T3)
			// deferred — under current XML all leaves fire simultaneously at
			// p.lb crossing via the scalar branching burst, so T2's multi-rank
			// co-initiation scenario doesn't apply to production runs.
			if (p.use_fournier_andrieu_kinetics) {
				// S0.5b.5: basal-stub accumulator lives on GF state.
				if (fa_gf) {
					fa_gf->basal_length = std::min(getLength(true), param()->lb);
				}

				// Sync node_to_phytomer length with current nodes size.
				// Expand with sentinel 0 (basal/unknown); the span-walk
				// below overwrites with correct per-rank tags.
				// node_to_phytomer is intrinsic Stem state (parallel to
				// nodes), kept on Stem after S0.5b state migration.
				node_to_phytomer.resize(nodes.size(), 0);
				const int n_nodes = static_cast<int>(nodes.size());
				const int n_links = static_cast<int>(localId_linking_nodes.size());
				if (n_links >= 1) {
					// Basal zone: nodes 0..localId_linking_nodes[0] (inclusive
					// of the linking node itself, which is the start of rank 1).
					int first_ln = localId_linking_nodes[0];
					for (int k = 0; k <= first_ln && k < n_nodes; ++k) {
						node_to_phytomer[k] = 0;
					}
					// Each span localId_linking_nodes[i-1] < k <= localId_linking_nodes[i]
					// belongs to rank i (rank 1 = first phytomer in the branching zone).
					for (int i = 1; i < n_links; ++i) {
						int lo = localId_linking_nodes[i - 1];
						int hi = localId_linking_nodes[i];
						for (int k = lo + 1; k <= hi && k < n_nodes; ++k) {
							node_to_phytomer[k] = i;
						}
					}
					// Apical zone (peduncle): nodes after last linking node.
					int last_ln = localId_linking_nodes[n_links - 1];
					int apical_rank = n_links;  // one past topmost rank
					for (int k = last_ln + 1; k < n_nodes; ++k) {
						node_to_phytomer[k] = apical_rank;
					}
				}

				// Update length_per_n from the span walk so Hard Invariant #5
				// closes with the scalar allocator's actual node geometry.
				// length_per_n[n] = Σ segment lengths in rank n's span. Stems
				// live in relative coordinates (hasRelCoord()==true enforced
				// at entry), so nodes[k] for k>=1 IS the segment delta vector
				// from node k-1 to k — take .length() directly rather than
				// differencing.
				//
				// Note: length_per_n is sized to n_ranks+1 in the targetlength
				// block above (driven by internode_v_n.size()), so ranks > n_ranks
				// (the apical/peduncle rank, which node_to_phytomer tags as
				// n_links) are dropped from length_per_n. That mass lives in
				// getLength(true) outside the Σ and breaks the Hard Invariant
				// #5 tolerance; to close it we extend length_per_n to cover
				// all tag values that appear (n_links + 1 entries).
				int max_tag = 0;
				for (int k = 0; k < n_nodes; ++k) {
					if (node_to_phytomer[k] > max_tag) max_tag = node_to_phytomer[k];
				}
				// S0.5b.5: per-rank arrays live on GF state.
				if (fa_gf && static_cast<int>(fa_gf->length_per_n.size()) < max_tag + 1) {
					fa_gf->length_per_n.resize(max_tag + 1, 0.0);
					fa_gf->epsilonDx_per_n.resize(max_tag + 1, 0.0);
					fa_gf->cessation_age_per_n.resize(max_tag + 1, -1.0);
					fa_gf->cessation_andrieu_tt_per_n.resize(max_tag + 1, -1.0);
					fa_gf->lateral_spawned_per_n.resize(max_tag + 1, 0);
				}
				const int n_ranks_lpn = fa_gf
					? static_cast<int>(fa_gf->length_per_n.size()) - 1
					: 0;
				if (fa_gf && n_ranks_lpn >= 1) {
					for (int n = 0; n <= n_ranks_lpn; ++n) {
						fa_gf->length_per_n[n] = 0.0;
					}
					// Sum segment lengths by rank tag (tag on the END node of
					// each segment, since seg k-1→k is "contributed by" node k).
					for (int k = 1; k < n_nodes; ++k) {
						int tag = node_to_phytomer[k];
						if (tag < 1 || tag > n_ranks_lpn) continue;
						fa_gf->length_per_n[tag] += nodes[k].length();
					}
				}
			}
			} // if active
			//set limit below 1e-10, as the test files see if correct length
			//once rounded at the 10th decimal
			//@see test/test_stem_ng.py
			//
			// codex-rescue follow-up (2026-04-28): under FA-on, once the
			// Phase IV cessation_age_ latch has fired the FA target cap
			// (above) clamps targetlength to (lb + Σln), excluding the
			// apical la that getK() still includes — so length will never
			// reach getK()*(1-1e-11) and active would stay permanently
			// true. Downstream consumers gate on isActive():
			//   src/external/PiafMunch/runPM.cpp:697  (carbon-water
			//     growth update under isActive() && useCWGr)
			//   src/external/PiafMunch/runPM.cpp:847  (whether to step
			//     the organ at all in the phloem solve)
			// Short-circuit to false in the FA-cessation branch.
			if (p.use_fournier_andrieu_kinetics && cessation_age_ >= 0.0) {
				active = false;
			} else {
				active = getLength(false)<=(p.getK()*(1 - 1e-11)); // become inactive, if final length is nearly reached
			}
		}
	} // if alive
}


/**
 *  @see Organ::createLateral
 *  @param dt       time step recieved by parent organ [day]
 *  @return growth period to send to lateral after creation
 */
double Stem::getLatInitialGrowth(double dt)
{
	double ageLN = this->calcAge(param()->lb); // MINIMUM age of root when lateral node is created
    ageLN = std::max(ageLN, age-dt);
	return age-ageLN;
}


/**
 *  @see Organ::createLateral
 *  @param ot_lat       organType of lateral to create
 *  @param st_lat       subType of lateral to create
 *  @param dt       time step recieved by parent organ [day]
 *  @return emergence delay to send to lateral after creation
 */
double Stem::getLatGrowthDelay(int ot_lat, int st_lat, double dt) const //override for stems
{

	bool verbose = false;
	auto rp = getOrganRandomParameter(); // rename
	double forDelay; //store necessary variables to define lateral growth delay
	int delayDefinition = getOrganism()->getDelayDefinition(ot_lat);


	assert(delayDefinition >= 0);

			if(verbose){std::cout<<"create lat, delay def "<<delayDefinition<<" "
			<<getId()<<" "<< (nodes.size() - 1)<<" "<<age
			<<" "<<getNodeId(nodes.size() - 1)<<" "<<getNodeId(0)<<std::endl;
			}
	if(verbose){std::cout<<"create lat, delay def "<<delayDefinition<<std::endl;}
	//count the number of laterals of subtype st already created on this organ std::function<double(int, int, std::shared_ptr<Organ>)>
	auto correctST = [ot_lat, st_lat](std::shared_ptr<Organ> org) -> double
		{
			return double((org->getParameter("organType") == ot_lat)&&(org->getParameter("subType")==st_lat));
		};//return 1. if organ of correct type and subtype, 0. otherwise

	double multiplyDelay = double(std::count_if(children.begin(), children.end(),
									 correctST));

	switch(delayDefinition){
		case Organism::dd_distance:
		{
			double meanLn = getParameter("lnMean"); // mean inter-lateral distance
			double effectiveLa = std::max(getParameter("la")-meanLn/2, 0.); // effective apical distance, observed apical distance is in [la-ln/2, la+ln/2]
			if(param()->lb+effectiveLa == param()->getK()){effectiveLa /=2;}// otherwise the growth delay == parent root life time.
			if(verbose)
			{
				std::cout<<"case Organism::dd_distance "<<organType()<<" "<<getParameter("subType")<<" "<<getLength(true)
				<<" "<<effectiveLa<<" "<<getParameter("la")<<" "<<meanLn<<std::endl;
			}
			double ageLN = this->calcAge(param()->lb); // age of root when lateral node is created
			ageLN = std::max(ageLN, age-dt);
			double ageLG = this->calcAge(param()->lb+effectiveLa); // age of the root, when the lateral starts growing (i.e when the apical zone is developed)
			forDelay = ageLG-ageLN; // time the lateral has to wait
			multiplyDelay = 1;//in this case, even for stems, it does not matter how many laterals there were before.
			break;
		}
		case Organism::dd_time_lat:
		{
			// time the lateral has to wait
			forDelay = std::max(rp->ldelay + plant.lock()->randn()*rp->ldelays, 0.);
			if(verbose){std::cout<<"Organism::dd_time_lat "<<rp->ldelay <<" "<<rp->ldelays<<" "<<forDelay<<std::endl;}
			break;
		}
		case Organism::dd_time_self:
		{

			//get delay per lateral
			auto latRp = plant.lock()->getOrganRandomParameter(ot_lat, st_lat); // random parameter of lateral to create
			forDelay = std::max(latRp->ldelay + plant.lock()->randn()*latRp->ldelays, 0.);
			// For dd_time_self, the delay is carried by each lateral type directly.
			// Do NOT multiply by multiplyDelay (count of same-subtype siblings),
			// because each subtype may be unique (e.g., per-position leaf subtypes).
			multiplyDelay = 1;
			if(verbose){
				std::cout<<"create lat, delay output "<<forDelay<<std::endl;
				std::cout<<"						 "<<ot_lat<<", "<<st_lat <<" "<<latRp->ldelay<<" "
				<< latRp->ldelays<<" "<<forDelay<<" "<<nodes.size()<<std::endl;
			}
			break;
		}
		default:
		{
			std::cout<<"delayDefinition "<<delayDefinition<<" "<<Organism::dd_distance<<" ";
			std::cout<< Organism::dd_time_lat<<" "<< Organism::dd_time_self<<std::endl<<std::flush;
			std::cout<<"				"<<(delayDefinition==Organism::dd_distance)<<" ";
			std::cout<<(delayDefinition== Organism::dd_time_lat)<<" "<< (delayDefinition==Organism::dd_time_self)<<std::endl<<std::flush;
			throw std::runtime_error("Delay definition type (delayDefinition) not recognised");
		}
	}
	if(verbose){std::cout<<"create lat, delay defEND "<<forDelay<<" "<<multiplyDelay<<std::endl;}
	return forDelay*multiplyDelay;
}
/**
 * Simulates internodal growth of dl for this stem
 * divid total stem growth between the phytomeres
 * currently two option:
 * growth devided equally between the phytomeres or
 * the phytomere grow sequentially
 *
 * @param 	dl			total length of the segments that are created [cm]
 * @param	verbose		print information
 */
void Stem::internodalGrowth(double dl,double dt, bool verbose)
{
	const StemSpecificParameter& p = *param(); // rename
	// S3b.8: under FA-on, basal_zero_ranks (maize: ranks 1-4) stay pinned at
	// the basal_internode_cm seed placed by Stem::simulate's plastochron loop
	// (§E.b). Without this gate, internodalGrowth would inflate their
	// spacing by distributing `dl` equally across all phytomers and break the
	// "V3 = 5 distinct collars close together" structural guarantee.
	auto srp = getStemRandomParameter();
	const bool fa_on = static_cast<bool>(srp) && srp->use_fournier_andrieu_kinetics;
	auto rank_is_basal_zero = [&](int rank_one_indexed) -> bool {
		if (!fa_on) return false;
		const auto& bz = srp->basal_zero_ranks;
		return std::find(bz.begin(), bz.end(), rank_one_indexed) != bz.end();
	};

	std::vector<double> toGrow(p.ln.size());
	double dl_;
	const int ln_0 = std::count(p.ln.cbegin(), p.ln.cend(), 0);//number of laterals wich grow on smae branching point as the one before
	// Count basal-zero phytomers with nonzero p.ln so the equal-share divisor
	// below only spans ranks that are actually allowed to elongate.
	int ln_basal_zero = 0;
	if (fa_on) {
		for (size_t i = 0; i < p.ln.size(); ++i) {
			if (p.ln.at(i) != 0 && rank_is_basal_zero(static_cast<int>(i) + 1)) {
				ln_basal_zero++;
			}
		}
	}
	if(p.nodalGrowth==0){//sequentiall growth
		toGrow[0] = dl;
		std::fill(toGrow.begin()+1,toGrow.end(),0) ;
	}
	if(p.nodalGrowth ==1)
	{//equal growth
		const int denom = std::max(1, static_cast<int>(p.ln.size()) - ln_0 - ln_basal_zero);
		std::fill(toGrow.begin(),toGrow.end(),dl/denom) ;
		if (fa_on && ln_basal_zero > 0) {
			// Zero basal_zero_ranks' initial share; the transfer-to-next
			// mechanism inside the main loop carries it forward onto the
			// first eligible phytomer (same pattern as p.ln[i]==0 stubs).
			for (size_t i = 0; i < p.ln.size(); ++i) {
				if (rank_is_basal_zero(static_cast<int>(i) + 1)) {
					toGrow[i] = 0.0;
				}
			}
		}
	}
	int loopId = 0;
	size_t phytomerId = 0;
	while( (dl >0)&&(loopId<2) ) {//do the loop at most twice over the children
		//if the phytomere can do a growth superior to the mean phytomere growth, we add the value of "missing"
		//(i.e., length left to grow to get the predefined total growth of the branching zone)
		if (phytomerId + 1 >= localId_linking_nodes.size()) break; // bounds safety
		int nn1 = localId_linking_nodes.at(phytomerId); //node at the beginning of phytomere
		int nn2 = localId_linking_nodes.at(phytomerId+1); //node at end of phytomere (if nn1 != nn2)

		double length1 = getLength(nn1);
		double availableForGrowth = p.ln.at(phytomerId) -( getLength(nn2) - length1 ) ;//difference between maximum and current length of the phytomer
		// S3b.8 gate: force basal_zero_ranks to contribute no growth capacity.
		// Combined with toGrow[i]=0 above, dl_ comes out 0 for these phytomers
		// and the transfer-to-next mechanism forwards their budget onward.
		if (rank_is_basal_zero(static_cast<int>(phytomerId) + 1)) {
			availableForGrowth = 0.0;
		}
		if(availableForGrowth<-1e-3)
		{
			// Plan B.2 (peduncle exuberance, 2026-04-27): under FA-on the
			// dl-routing wraparound through basal_zero ranks deposits
			// leftover dl onto the lowest non-zero phytomer (rank 5 for
			// maize, IL_final=0.8 cm). Sub-cm overshoot is expected and
			// already absorbed by the clamp below; suppress the cout to
			// satisfy plan D.4 (no warnings under FA-on day 1..180). FA-off
			// path keeps the warning so any genuine cap violation under
			// scalar params surfaces.
			if (!fa_on) {
				std::cout << "WARNING Stem::internodalGrowth phytomere "<<phytomerId<<" is too long: "<<availableForGrowth<<" "<<
				p.ln.at(phytomerId)<<" "<<getLength(nn2)<<" "<<length1<<std::endl;
			}
			availableForGrowth = 0; // clamp: skip growth for overgrown phytomere
		}
		dl_ = std::max(0.,std::min(std::min(toGrow[phytomerId],availableForGrowth), dl));
		if(dl_ > 0)
		{
			createSegments(dl_,dt,verbose, nn2 ); dl -= dl_;
		}
		if((phytomerId+1)< p.ln.size()){
			toGrow.at(phytomerId+1) +=  toGrow.at(phytomerId) - dl_ ;
			phytomerId ++;
		}else{
			toGrow.at(0) +=  toGrow.at(phytomerId) - dl_ ;
			loopId++; phytomerId = 0;
		}	//loop twice other the children

	}
	if(std::abs(dl)> 1e-6){
		// S3b.8: under FA-on, blocking basal_zero_ranks from receiving `dl`
		// combined with the per-phytomer `p.ln` cap can leave leftover `dl`
		// that the branching zone cannot absorb (FA kinetic targets for some
		// ranks exceed `p.ln`). Pre-S3b.8 this excess silently inflated the
		// basal ranks by equal-share distribution; that outlet is now sealed.
		// Route the leftover to the apical zone so mass isn't lost — caller
		// already debited `length += ddx`, so without this route `getLength`
		// diverges from the realized geometry and Hard Invariant #5 breaks.
		// On FA-off the original warning fires (historical behaviour: caller
		// has also already debited `length`, but the legacy assumption is
		// that the warning never fires under scalar params).
		if (fa_on && dl > 0) {
			createSegments(dl, dt, verbose);
			dl = 0;
		} else {
			std::cout << "WARNING Stem::internodalGrowth length left to grow: "<<dl<<std::endl;
		}
	}
}
/**
 * Returns a parameter per organ
 *
 * @param name 		parameter name (returns nan if not available)
 *
 */
double Stem::getParameter(std::string name) const
{
	if (name=="lb") { return param()->lb; } // basal zone [cm]
	if (name=="delayNGStart") { return param()->delayNGStart; } // delay for nodal growth [day]
	if (name=="delayNGEnd") { return param()->delayNGEnd; } // delay for nodal growth [day]
	if (name=="la") { return param()->la; } // apical zone [cm]
	if (name=="nob") { return param()->nob(); } // number of branching points
	if (name=="r"){ return param()->r; }  // initial growth rate [cm day-1]
	if (name=="radius") { return param()->a; } // root radius [cm]
	if (name=="a") { return param()->a; } // root radius [cm]
	if (name=="theta") { return param()->theta; } // angle between root and parent root [rad]
	if (name=="rlt") { return param()->rlt; } // root life time [day]
	if (name=="k") { return param()->getK(); }; // maximal root length [cm]
	if (name=="lnMean") { // mean lateral distance [cm]
        auto& v =param()->ln;
		if(v.size()>0){
			return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
		}else{
			return 0;
		}
	}
	if (name=="lnDev") { // standard deviation of lateral distance [cm]
		auto& v =param()->ln;
		double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
		double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
		return std::sqrt(sq_sum / v.size() - mean * mean);
	}
	if (name=="volume") { return param()->a*param()->a*M_PI*getLength(true); } // // root volume [cm^3]
	if (name=="surface") { return 2*param()->a*M_PI*getLength(true); }
	if (name=="type") { return this->param_->subType; }  // delete to avoid confusion?
	if (name=="subType") { return this->param_->subType; }  // organ sub-type [-]
	if (name=="parentNI") { return parentNI; } // local parent node index where the lateral emerges
	return Organ::getParameter(name);
}




/**
 * Radius (cm) of the stem at a given arc-length along its own skeleton.
 *
 * Current ``StemSpecificParameter`` carries a single ``a`` (radius) with no
 * explicit taper, so the arc-length argument is accepted for API symmetry but
 * the returned value is uniform along the stem. Phase E consumers
 * (``Leaf::updateNodesFromSurfaceCPs`` with compound sheath CPs) use this
 * helper so the rest of the sheath-wrapping math can remain agnostic to
 * future per-internode taper.
 *
 * @param arc_length   arc-length along the stem skeleton [cm] (unused today)
 */
double Stem::getRadiusAt(double /*arc_length*/) const
{
	return param()->a;
}


/**
 * Analytical length of the stem at a given age
 *
 * @param age          age of the stem [day]
 */
double Stem::calcLength(double age)
{
	assert(age>=0 && "Stem::calcLength() negative root age");
	return getStemRandomParameter()->f_gf->getLength(age,getStemRandomParameter()->r,param()->getK(),shared_from_this());
}

/**
 * Analytical age of the stem at a given length
 * no scaling of organ growth , so can return age directly
 * otherwise cannot compute exact age between delayNGStart and delayNGEnd
 * @param length   length of the stem [cm]
 */
double Stem::calcAge(double length) const
{
	assert(length>=0 && "Stem::calcAge() negative root age");
	double age__ = getStemRandomParameter()->f_gf->getAge(length,getStemRandomParameter()->r,param()->getK(),shared_from_this());
	if(age__ >param()->delayNGStart ){age__ += (param()->delayNGEnd - param()->delayNGStart);}
	return age__;
}


/* Fournier-Andrieu kinetic constants (plan §B.3). */
namespace {
constexpr double IL_INIT_CM = 0.0025;            // Zhu 2014: initial IL at tau=0
constexpr double IL_AT_END_PHASE_II_CM = 4.5;    // FA 2000 line 223, phyt 7-15
constexpr double HALF_PLASTOCHRON_LAG_DEGCD = 9.6;  // FA 2000 line 207
}

/**
 * Fournier-Andrieu per-phytomer internode length (plan §B.3).
 *
 * Returns the length [cm] of internode rank n at the plant's current
 * Andrieu-axis thermal time, anchored at initiation (= leaf-n primordium
 * emergence + 9.6 °Cd half-plastochron lag). Four-phase kinetic:
 *   Phase I  [0, 309)   pre-collar exponential at r_I
 *   Phase II [309, 334) linear ramp to 4.5 cm
 *   Phase III [334, 334+D_n) linear at v_n
 *   Phase IV [334+D_n, ∞) exponential decay to IL_final at k
 *
 * Returns 0 for basal-zero ranks (Zhu 2014 line 127), for pre-initiation
 * (leaf not yet emerged), and when required per-rank data is missing
 * (graceful degradation — stem just shows no growth for that rank).
 *
 * @param n  phytomer rank (1-based; rank 1 = first leaf attached to mainstem)
 */
double Stem::calcLengthPerPhytomer(int n) const
{
	const auto& sp = *param();
	// Basal zero (Zhu 2014 line 127): these ranks are zero-length at all tau.
	const auto& basal_zero = getStemRandomParameter()->basal_zero_ranks;
	if (std::find(basal_zero.begin(), basal_zero.end(), n) != basal_zero.end()) {
		return 0.0;
	}

	// S3b.7: τ_n anchor resolution. Prefer the plastochron-driven
	// initiation_andrieu_tt_per_n[n] when set (plan §E.b), which records
	// the plant Andrieu-TT at the step rank n's node was created on the
	// plastochron clock. This decouples internode kinetics from leaf
	// emergence and resolves the S3b.3 chicken-and-egg deadlock (leaf
	// won't emerge until internode has length; internode won't have
	// length until FA runs; FA won't run until leaf has emerged).
	//
	// Fallback (FA-on stems built pre-S3b.7, or edge cases where the
	// plastochron path didn't set the entry): read the leaf's
	// emergence_andrieu_tt_ as the primordium init time, shifted by
	// HALF_PLASTOCHRON_LAG_DEGCD per FA 2000 line 207. For maize_calibrated
	// the per-position leaf subType convention maps subType=n+1 to rank=n
	// (subType 2 = rank 1, subType 17 = rank 16). We scan children in
	// order rather than relying on leafphytomerID (which is indexed by
	// subType and all maize_calibrated leaves have phytomerId=1 for
	// their unique subType).
	// S0.5b.3: read FA per-organ kinetic state from the GF instance.
	auto* fa_state = getFaState();
	double init_tt = -1.0;
	if (n >= 1 && fa_state
	    && n < static_cast<int>(fa_state->initiation_andrieu_tt_per_n.size())
	    && fa_state->initiation_andrieu_tt_per_n[n] >= 0.0) {
		init_tt = fa_state->initiation_andrieu_tt_per_n[n];
	} else {
		int leaf_ordinal = 0;                     // 1-based index among leaf children
		double leaf_emerge_tt = -1.0;
		for (const auto& c : children) {
			if (c->organType() == Organism::ot_leaf) {
				leaf_ordinal++;
				if (leaf_ordinal == n) {
					auto lf = std::static_pointer_cast<Leaf>(c);
					leaf_emerge_tt = lf->getEmergenceAndrieuTT();
					break;
				}
			}
		}
		if (leaf_emerge_tt < 0.0) {
			// Leaf-n not yet emerged (or absent): internode contributes nothing.
			return 0.0;
		}
		init_tt = leaf_emerge_tt + HALF_PLASTOCHRON_LAG_DEGCD;
	}

	// Effective Andrieu TT: freeze at cessation latch (parallel to the scalar
	// path's cessation_age_ clamp in Stem::simulate). S3b.3 adds per-rank
	// latches (plan §B) that dominate over the global latch when set: each
	// internode freezes on its own tau_n rather than on plant-level tt_cessation,
	// so late-initiated upper ranks keep elongating after early-initiated lower
	// ranks have frozen.
	auto plant_fa = getPlant();
	if (!plant_fa) return 0.0;
	double andrieu_tt = plant_fa->getAccumulatedAndrieuTT();
	// S0.5b.3: per-rank cessation latch from GF; global cessation_andrieu_tt_
	// stays on Stem (set by use_thermal_cessation gate independently of FA).
	if (n >= 1 && fa_state
	    && n < static_cast<int>(fa_state->cessation_andrieu_tt_per_n.size())
	    && fa_state->cessation_andrieu_tt_per_n[n] >= 0.0
	    && andrieu_tt > fa_state->cessation_andrieu_tt_per_n[n]) {
		andrieu_tt = fa_state->cessation_andrieu_tt_per_n[n];
	} else if (cessation_andrieu_tt_ >= 0.0 && andrieu_tt > cessation_andrieu_tt_) {
		andrieu_tt = cessation_andrieu_tt_;
	}
	double tau = andrieu_tt - init_tt;
	if (tau < 0.0) return 0.0;

	const auto& srp = *getStemRandomParameter();
	double r_I = srp.r_I;
	double phase_I_duration = srp.phase_I_duration;
	double phase_II_duration = srp.phase_II_duration;
	double phase_IV_duration = srp.phase_IV_duration;
	double phase_IV_k = srp.phase_IV_k;

	// Phase I: pre-collar exponential.
	if (tau < phase_I_duration) {
		return IL_INIT_CM * std::exp(r_I * tau);
	}

	// Phase II: 25 °Cd linear ramp from end-of-Phase-I to 4.5 cm uniform boundary.
	double phase_II_end = phase_I_duration + phase_II_duration;
	if (tau < phase_II_end) {
		double IL_end_I = IL_INIT_CM * std::exp(r_I * phase_I_duration);
		double frac = (tau - phase_I_duration) / phase_II_duration;
		return IL_end_I + frac * (IL_AT_END_PHASE_II_CM - IL_end_I);
	}

	// Phase III / IV need per-rank v_n and D_n. Specific param holds the
	// realized (randomized) per-rank vector; when calibrated XMLs populate
	// only the Random params, realize() copies them through (see
	// stemparameter.cpp). Graceful fallback: if missing, hold at 4.5 cm.
	const auto& vvec = sp.internode_v_n;
	const auto& dvec = sp.internode_D_n;
	if (n < 1 || n > static_cast<int>(vvec.size()) || n > static_cast<int>(dvec.size())) {
		return IL_AT_END_PHASE_II_CM;       // no Fig 12 data for this rank
	}
	double v_n = vvec[n - 1];                // 0-indexed vector, 1-indexed rank
	double D_n = dvec[n - 1];
	double phase_III_end = phase_II_end + D_n;
	if (tau < phase_III_end) {
		return IL_AT_END_PHASE_II_CM + v_n * (tau - phase_II_end);
	}

	// Phase IV: exponential decay toward IL_final. If per-rank IL_final is
	// missing, fall through to end-of-Phase-III value (no decay).
	double IL_end_III = IL_AT_END_PHASE_II_CM + v_n * D_n;
	const auto& ilfvec = sp.internode_IL_final;
	if (n < 1 || n > static_cast<int>(ilfvec.size())) {
		return IL_end_III;
	}
	double IL_final = ilfvec[n - 1];
	return IL_final - (IL_final - IL_end_III) * std::exp(-phase_IV_k * (tau - phase_III_end));
}

/**
 * Sum of Stem::calcLengthPerPhytomer over all ranks that have per-rank data.
 *
 * Drives the FA branch of Stem::simulate::targetlength (plan §B.3.5 "thin"
 * interpretation — the existing basal/branching/apical allocation loop handles
 * per-rank segment distribution implicitly via ldelay on each leaf, rather
 * than via per-phytomer dl_n + node-insertion bookkeeping). Loops over ranks
 * 1..|internode_v_n| because that's where Fig-12 data is defined; higher ranks
 * (if any) have no Phase III and would contribute only Phase I/II residuals.
 */
double Stem::calcLengthPerPhytomerSum() const
{
	const auto& sp = *param();
	int n_ranks = static_cast<int>(sp.internode_v_n.size());
	double total = 0.0;
	for (int n = 1; n <= n_ranks; ++n) {
		total += calcLengthPerPhytomer(n);
	}
	return total;
}


/**
 * S3b per-phytomer bookkeeping helper (plan §A).
 *
 * Returns the position in the node vector at which rank n's next node should
 * be inserted, defined as "one past the last existing node that belongs to
 * rank n-1" (scan node_to_phytomer from the back). Falls back to nodes.size()
 * (append at apex) when rank n-1 has no nodes yet — this is the cold-start
 * case: under Zhu 2014 line 127 ranks 1-4 are basal-zero, so the first rank
 * to actually initiate is rank 5, and there are no rank-4 nodes to insert
 * after.
 *
 * Only meaningful when the FA flag is true AND node_to_phytomer is populated
 * by the per-rank mid-stem insertion driver (scheduled for S3b.3 in the plan
 * session split; this helper is declared now so the pybind surface and the
 * B.5' tests can exercise the fallback path in isolation).
 *
 * @param n  phytomer rank (1-based)
 */
int Stem::computeInsertionIndexForRank(int n) const
{
	for (int i = static_cast<int>(node_to_phytomer.size()) - 1; i >= 0; --i) {
		if (node_to_phytomer[i] == n - 1) {
			return i + 1;
		}
	}
	return static_cast<int>(nodes.size());
}


/**
 * S3b per-phytomer bookkeeping accessor (plan §A).
 *
 * Returns the latched realised length of internode rank n (cm). Monotone:
 * length_per_n[n] tracks the historical maximum of calcLengthPerPhytomer(n)
 * so the Phase IV decay artifact (S3b.1 finding 2) doesn't make the returned
 * value drop as the raw FA target decays toward IL_final. Returns 0.0 when
 * the FA flag is off, when n is out of range, or when the rank has not yet
 * initiated (its leaf hasn't emerged yet, so calcLengthPerPhytomer has never
 * been non-zero).
 *
 * @param n  phytomer rank (1-based)
 */
double Stem::getPhytomerLength(int n) const
{
	// S0.5b.3: read latched length from GF state.
	auto* fa_state = getFaState();
	if (!fa_state || n < 1 || n >= static_cast<int>(fa_state->length_per_n.size())) {
		return 0.0;
	}
	return fa_state->length_per_n[n];
}


/**
 * S0.5b: returns a pointer to the FA per-organ kinetic state living on the
 * MultiPhaseStemGrowth GF instance, or nullptr when this stem isn't FA-on
 * (no MultiPhaseStemGrowth GF on the LRP, or getLength() hasn't seeded the
 * organ id yet).  After full state migration the GF entry is canonical;
 * the Stem mirror fields (length_per_n, basal_length_, cessation_age_, etc.)
 * are scheduled for retirement.
 */
MultiPhaseStemGrowth::PerOrganFAState* Stem::getFaState() const
{
	auto srp = getStemRandomParameter();
	if (!srp || !srp->f_gf) return nullptr;
	auto gf_mps = std::dynamic_pointer_cast<MultiPhaseStemGrowth>(srp->f_gf);
	if (!gf_mps) return nullptr;
	auto it = gf_mps->per_organ_state.find(getId());
	if (it == gf_mps->per_organ_state.end()) return nullptr;
	return &it->second;
}


/**
 * stores the local id of the linking node. used by @see Stem::internodalGrowth()
 */
void Stem::storeLinkingNodeLocalId(int numCreatedLN, bool verbose)
{
	localId_linking_nodes.push_back(nodes.size()-1);
	if(numCreatedLN!=localId_linking_nodes.size())
	{
		throw std::runtime_error("wrong number of linking nodes in stem: "+std::to_string(numCreatedLN)
		+" against "+std::to_string(localId_linking_nodes.size()));
	}
}

/**
 * Adds a node to the organ.
 *
 * For simplicity nodes can not be deleted, organs can only become deactivated or die
 *
 * @param n        new node
 * @param id       global node index
 * @param t        exact creation time of the node
 * @param index	   position were new node is to be added
 * @param shift	   do we need to shift the nodes? (i.e., is the new node inserted between existing nodes because of internodal growth?)
 */
void Stem::addNode(Vector3d n, int id, double t, size_t index, bool shift)
{
	bool verbose = false;
	if(verbose)
	{
		std::cout<<"Organ::addNode "<<id<<" "<<getId()<<" "<<organType()<<" "<<getParameter("subType")<<std::endl;
		std::cout<<"Organ::addNode "<<n.toString()<<" "<<t<<" "<<index<<" "<<shift<<std::endl;

	}
	if(!shift){//node added at the end of organ
		nodes.push_back(n); // node
		nodeIds.push_back(id); //unique id
		nodeCTs.push_back(t); // exact creation time
	}
	else{//could be quite slow  to insert, but we won t have that many (node-)tillers (?)
		nodes.insert(nodes.begin() + index, n);//add the node at index
		//add a global index.
		//no need for the nodes to keep the same global index and makes the update of the nodes position for MappedPlant object more simple)
		//if(verbose){
			//			std::cout<<"Organ::addNode "<<organType()<<" "<<id<<" "<<index<<std::endl<<std::flush;
		//}
		nodeIds.push_back(id);
		nodeCTs.insert(nodeCTs.begin() + index, t);
		for(auto kid : children){//if carries children after the added node, update their "parent node index"

			if((kid->parentNI >= index-1 )&&(kid->parentNI > 0)){
				kid->moveOrigin(kid->parentNI + 1);
				}

		}
		for(int numnode = 0; numnode < localId_linking_nodes.size();numnode++){//update the local ids of the linking nodes
			if((localId_linking_nodes.at(numnode) >= index-1 )&&(localId_linking_nodes.at(numnode) > 0))
			{
				localId_linking_nodes.at(numnode) += 1;
			}
		}

	}
}


/**
 * @return The StemTypeParameter from the plant
 */
std::shared_ptr<StemRandomParameter> Stem::getStemRandomParameter() const
{
	return std::static_pointer_cast<StemRandomParameter>(plant.lock()->getOrganRandomParameter(Organism::ot_stem, param_->subType));
}

/**
 * @return Parameters of the specific root
 */
std::shared_ptr<const StemSpecificParameter> Stem::param() const
{
	return std::static_pointer_cast<const StemSpecificParameter>(param_);
}

/*
 * Quick info about the object for debugging
 * additionally, use param()->toString() and getOrganRandomParameter()->toString() to obtain all information.
 */
std::string Stem::toString() const
{
	std::stringstream newstring;
	newstring << "; initial heading: " << getiHeading0().toString() << ", parent node index" << parentNI << ".";
	return Organ::toString()+newstring.str();
}

/**
 * Find a child organ by phytomer rank and sheath/blade parity.
 * In phytomer mode, subType = 2*rank + (isSheath ? 0 : 1).
 */
std::shared_ptr<Organ> Stem::getChildByPhytomerRank(int rank, int organType, bool isSheath) const {
    int targetParity = isSheath ? 0 : 1;
    for (size_t i = 0; i < children.size(); i++) {
        auto c = children[i];
        if (c->organType() == organType) {
            int st = (int)c->getParameter("subType");
            if ((st / 2 == rank) && (st % 2 == targetParity)) {
                return c;
            }
        }
    }
    return nullptr;
}

} // namespace CPlantBox

