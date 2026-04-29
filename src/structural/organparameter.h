// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#ifndef ORGANPARAMETER_H_
#define ORGANPARAMETER_H_

#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <vector>
#include <string>

#include "mymath.h"
#include "tinyxml2.h"

/**
 * This file describes the classes OrganSpecificParameter and OrganRandomParameter.
 * OrganSpecificParameter are drawn from the OrganRandomParameter class
 */

namespace CPlantBox {

class Organism; // forward declaration
class GrowthFunction;
class ExponentialGrowth;
class Tropism;

/**
 * S0.6 / Lock #1 / Lock #8 — axis flag for time-axis parameters.
 *
 * `delayNGEnd`, `ldelay`, and similar fields are scalar time thresholds that
 * historically lived on a calendar-day axis only. Lock #1 generalises them
 * by attaching an axis flag so the same field can carry a thermal-time
 * (Andrieu Tb=9.8 °Cd) threshold for stems gated on plant TT.
 *
 * `Calendar` is the default and matches the existing semantics
 * (bit-identical for every XML that does not specify `axis="..."`).
 * `TT` switches the consumer (`MultiPhaseStemGrowth::getLength`) to compare
 * against `Plant::getAccumulatedAndrieuTT()` instead of calendar-day age.
 *
 * The enum is shared between `delayNGEndAxis` (Lock #1, this ADR phase)
 * and the future `ldelayAxis` (Lock #8, leaf-side birth gate); putting
 * one enum on `OrganRandomParameter` keeps the upstream-PR surface coherent
 * (one biology-neutral generalisation, two consumers) and means non-stem
 * organ types pay zero footprint when they don't opt in.
 */
enum class DelayAxis { Calendar = 0, TT = 1 };

/**
 * Parameters for a specific organ
 */
class OrganSpecificParameter {
public:

    OrganSpecificParameter(int t, double a): subType(t), a(a)  { }

    virtual ~OrganSpecificParameter() { }
    int subType = -1; ///< sub type of the organ
    double a = 0.; ///< radius of the organ [cm]
    virtual std::string toString() const; ///< quick info for debugging

};

/**
 * Contains a parameter set describing as single sub type of an organ,
 * specific parameters are then created with realize().
 *
 * Organizes parameters in hash-maps for scalar double and scalar int values.
 * For this reason derived classes getParameter(), toString(), readXML(), and writeXML() should work out of the box.
 * For other parameter types the methods must be overwritten, see e.g. RootRandomParameter.
 *
 * The factory function copy() has to be overwritten for each specialization.
 */
class OrganRandomParameter
{
public:

    OrganRandomParameter(std::shared_ptr<Organism> plant); ///< default constructor
    virtual ~OrganRandomParameter() { };

    virtual std::shared_ptr<OrganRandomParameter> copy(std::shared_ptr<Organism> plant); ///< copies the root type parameter into a new plant

    virtual std::shared_ptr<OrganSpecificParameter> realize(); ///< creates a specific organ from the root parameter set

    virtual double getParameter(std::string name) const; // get a scalar parameter

    virtual std::string toString(bool verbose = true) const; ///< info for debugging

    virtual void readXML(tinyxml2::XMLElement* element, bool verbose); ///< reads a single sub type organ parameter set
	void readSuccessor(tinyxml2::XMLElement* p, bool verbose);
	void readXML(std::string name, bool verbose); ///< reads a single sub type organ parameter set
    virtual tinyxml2::XMLElement* writeXML(tinyxml2::XMLDocument& doc, bool comments = true) const; ///< writes a organ root parameter set
    void writeXML(std::string name) const; ///< writes a organ root parameter set

	int getLateralType(const Vector3d& pos, int ruleId); ///< Choose (dice) lateral type based on stem parameter set

	virtual void bindParameters(); ///<sets up class introspection
    void bindParameter(std::string name, int* i, std::string descr = "", double* dev = nullptr); ///< binds integer to parameter name
    void bindParameter(std::string name, double* d, std::string descr = "", double* dev = nullptr); ///< binds double to parameter name
    void bindAxisParameter(std::string name, DelayAxis* axis); ///< binds a DelayAxis flag to a scalar time-axis parameter (Lock #1 / Lock #8)

    std::string name = "organ";
    int organType = 0;
    int subType = 0;
    double a = 0.1; ///< Root radius [cm]
    double as = 0.; ///< Standard deviation root radius [cm]
    double dx = 0.25;///< Maximal segment size [cm]
	double dxMin = 1e-6; ///< threshold value, smaller segments will be skipped (otherwise stem tip direction can become NaN)
	double ldelay = -1.; ///< Lateral emergence delay [day], used by RootDelay, @see RootDelay, RootSystem::initializeDB or if Organism->delayDefinition != Organism::dd_distance
    double ldelays = 0.; ///< Standard deviation of lateral emergence delay [day]

    /* S0.6 / Lock #1: axis flag for delayNGEnd (which lives on
     * StemRandomParameter). When DelayAxis::TT, MultiPhaseStemGrowth
     * interprets `delayNGEnd` as an Andrieu-TT threshold rather than a
     * calendar-day delay. Default Calendar = bit-identical with every
     * existing XML; opt-in via `<parameter name="delayNGEnd" value="1500"
     * axis="TT"/>` per Lock #1 spec. The field lives on the base class so
     * the future ldelayAxis (Lock #8) can share the enum cleanly. */
    DelayAxis delayNGEndAxis = DelayAxis::Calendar;

    /* S0.8 / Lock #8: axis flag for `ldelay` (lateral emergence delay;
     * lives on this base class). When DelayAxis::TT, the Leaf::simulate
     * emergence gate (and, future-symmetric, the Stem::simulate gate)
     * interprets `ldelay` as an absolute Andrieu-TT threshold [degCd] on
     * the plant's accumulated TT axis instead of a calendar-day delay.
     * Default Calendar = bit-identical with every existing XML; opt-in
     * via `<parameter name="ldelay" value="<TT>" axis="TT"/>` per Lock #8
     * spec. Symmetric to Lock #1's delayNGEndAxis (cessation gate) — Lock
     * #8 is the birth-gate sibling that makes the upstream pitch
     * coherent. Eventually retires `tt_emergence` + `use_thermal_emergence`
     * as parallel fields once consumers and XMLs migrate. */
    DelayAxis ldelayAxis = DelayAxis::Calendar;
    std::vector<std::vector<double> > successorWhere = std::vector<std::vector<double>>(0, std::vector<double> (0, 0));
    ///< Where should rule be implemented [1] or not [-1]; need to use double to distiguish between -0 and 0; default: vector empty == rule implemented everywhere
    std::vector<std::vector<int> > successorOT = std::vector<std::vector<int>>(0, std::vector<int> (0, 0)); ///< Lateral types [1]
    std::vector<std::vector<int> > successorST = std::vector<std::vector<int>>(0, std::vector<int> (0, 0)); ///< Lateral types [1]
    std::vector<std::vector<double>> successorP = std::vector<std::vector<double>>(0, std::vector<double> (0, 0)); ///< Probabilities of lateral type to emerge (sum of values == 1) [1]
    std::vector<int>  successorNo = std::vector<int>(0); ///< Lateral types [1]

    std::weak_ptr<Organism> plant;
	std::shared_ptr<Tropism> f_tf;  ///< tropism function (defined in constructor as new Tropism(plant))
    std::shared_ptr<GrowthFunction> f_gf;


protected:

    /* class introspection */
    std::map<std::string, double*> dparam; ///< Parameters with type double that can be read and written
    std::map<std::string, int*> iparam; ///< Parameters with type double that can be read and written
    std::map<std::string, double*> param_sd; ///< Deviations of parameters
    std::map<std::string, std::string> description; ///< Parameter descriptions
    std::map<std::string, DelayAxis*> axis_param; ///< Lock #1 / Lock #8: axis flags attached to scalar time-axis parameters

    std::string vector2string(std::vector<int> vec) const;
    std::string vector2string(std::vector<double> vec) const;
    std::vector<int> string2vector(const char* xmlInput, int defaultVal);
    std::vector<double> string2vector(const char* xmlInput, double defaultVal);

    template <class IntOrDouble>
    void cpb_queryStringAttribute(std::vector<std::string> keyNames,IntOrDouble defaultVal,int sizeVector,
                                    bool replaceByDefault,
                                    std::vector<IntOrDouble> & vToFill, tinyxml2::XMLElement* key);


};

} // namespace

#endif
