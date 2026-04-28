#ifndef PLANT_H_
#define PLANT_H_

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <random>
#include <numeric>
#include <istream>

#include "Organ.h"
#include "Root.h"
#include "Seed.h"
#include "Stem.h"
#include "Leaf.h"

#include "soil.h"
#include "tropism.h"
#include "growth.h"
#include "tinyxml2.h"

namespace CPlantBox {

/**
 * Plant
 *
 * This class manages all model parameters, the simulation,
 * and stores the seed of the plant,
 * and offers utility functions for post processing.
 *
 * The Plant class inherits from Organism and provides additional functionality:
 * - Manages the OrganRandomParameters
 * - Offers an interface for the simulation loop (initialize, simulate, ...)
 * - Collects node and line segment geometry from the organ tree
 * - Collect parameters from the organs
 * - Can collect information about the last time step
 * - Supports RSML
 * - Holds global node index and organ index counter
 * - Holds random numbers generator for the organ classes
 *
 */
class Plant :public Organism
{
public:

  enum TropismTypes { tt_plagio = 0, tt_gravi = 1, tt_exo = 2, tt_hydro = 3, tt_antigravi = 4, tt_twist = 5,  tt_antigravi2gravi = 6};  ///< plant tropism types
  enum GrowthFunctionTypes { gft_negexp = 1, gft_linear = 2 , gft_CWLim = 3, gft_gompertz = 4, gft_multi_phase_stem = 5 }; // plant growth function

  Plant(unsigned int seednum  = 0.);
  virtual ~Plant() { };

  std::shared_ptr<Organism> copy() override; ///< deep copies the organism

  /* parameters */
  void initializeReader() override; ///< initializes XML reader
  void readParameters(std::string name, std::string  basetag = "plant", bool fromFile = true, bool verbose = true) override {this->initializeReader(); Organism::readParameters(name, basetag, fromFile, verbose); };
  void openXML(std::string name) { readParameters(name); } // old name
  std::shared_ptr<SeedRandomParameter> getSeedRandomParameter();

  /* Simulation */
  void setSoil(std::shared_ptr<SoilLookUp> soil_) { soil = soil_; } ///< optionally sets a soil for hydro tropism (call before Plant::initialize())
  void reset(); ///< resets the plant class, keeps the organ type parameters
  virtual void initializeLB(bool verbose = true); ///< creates the base roots (length based lateral emergence times), call before simulation and after setting plant and root parameters
  virtual void initializeDB(bool verbose = true); ///< creates the base roots (delay based lateral emergence times), call before simulation and after setting plant and root parameters
  void initialize(bool verbose = true) override { initializeLB(verbose); };
  void setTropism(std::shared_ptr<Tropism> tf, int organType, int subType = -1); ///< todo docme
  void simulate(); ///< simulates root system growth for the time defined in the root system parameters
  void simulate(double dt, bool verbose = false) override;
  void simulate(double dt, double maxinc, std::shared_ptr<ProportionalElongation> se, bool verbose = true); ///< simulates the plant with a maximal elongation
  void simulateLimited(double dt, double max_, std::string paramName, std::vector<double> scales, std::shared_ptr<ProportionalElongation> se, bool verbose);  ///< simulates plant with limited costs

  /* call back function creation */
  void initCallbacks(); ///< sets up callback functions for tropisms and growth functions, called by initialize()
  std::shared_ptr<Tropism> createTropismFunction(int tt, double N, double sigma, double Tage = 0.); ///< Creates the tropisms, overwrite or change this method to add more tropisms
  virtual std::shared_ptr<GrowthFunction> createGrowthFunction(int gft); ///< Creates the growth function per root type, overwrite or change this method to add more tropisms

  std::string toString() const override;

  std::vector<int> leafphytomerID = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  void abs2rel();
  void rel2abs();

  /* Air temperature for thermal-time elongation (Step 2B) */
  void setAirTemperature(double T) { airTemperature_ = T; }
  double getAirTemperature() const { return airTemperature_; }

  /* Plant-level thermal-time accumulator (real degCd, T_base-clamped) */
  double getAccumulatedTT() const { return accumulatedTT_; }
  void setAccumulatedTT(double tt) { accumulatedTT_ = tt; }
  void setCardinalTemperatures(double T_base, double T_opt, double T_max) {
      tt_T_base_ = T_base; tt_T_opt_ = T_opt; tt_T_max_ = T_max;
  }
  double getCardinalTBase() const { return tt_T_base_; }

  /* Dual-axis Andrieu TT accumulator (Tb=9.8 °C) for Fournier-Andrieu internode
   * kinetics. Runs alongside accumulatedTT_ (Tb=8.0). See
   * PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §B.2. */
  double getAccumulatedAndrieuTT() const { return andrieu_tt_; }
  void setAccumulatedAndrieuTT(double tt) { andrieu_tt_ = tt; }
  void setAndrieuTBase(double T_base) { tt_T_base_andrieu_ = T_base; }
  double getAndrieuTBase() const { return tt_T_base_andrieu_; }

protected:

  double airTemperature_ = 25.0;  ///< current air temperature [deg C]
  double accumulatedTT_ = 0.0;    ///< plant thermal time [deg Cd], real units
  double tt_T_base_ = 8.0;        ///< maize default
  double tt_T_opt_  = 30.0;
  double tt_T_max_  = 41.0;

  /* Andrieu-axis TT accumulator (F-A 2000 + Birch 2002 kinetic Tb). Only
   * consumed by FA-path consumers (Stem::simulate when
   * use_fournier_andrieu_kinetics=true). Unused by any existing code path —
   * always-on accumulation is harmless bit-for-bit (no one reads it yet). */
  double andrieu_tt_ = 0.0;           ///< plant thermal time on Andrieu axis [deg Cd]
  double tt_T_base_andrieu_ = 9.8;    ///< Andrieu axis base temperature [deg C]

  std::shared_ptr<SoilLookUp> soil; ///< callback for hydro, or chemo tropism (needs to set before initialize()) TODO should be a part of tf, or rtparam

  void initialize_(bool verbose = true); // called by initializeLB, and initializeDB
  double weightedSum(std::string paramName, std::vector<double> scales) const; // weighted sum per organ type (used by simulate_limited)

};

} // namespace CPlantBox

#endif /* PLANT_H_ */
