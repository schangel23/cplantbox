// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#include "leafshape_distribution.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace CPlantBox {

namespace {

// ============================================================
// Minimal JSON reader — scoped to the maize_leaf_shape_distribution.json
// schema. We do not need a general-purpose parser; the schema is fixed at
// bake time by S0's fitter and the only string values we read are the
// integer-valued rank keys ("0" .. "14"). Other string fields
// (frame_convention, schema_version, ...) are skipped.
//
// Adding a third-party JSON dependency is out of scope here: see the
// S4 commit message for the rationale (single-purpose, ~250 LOC, no new
// external surface). If a second consumer of JSON appears in the
// codebase, switch to nlohmann/json single-header at that point.
// ============================================================

class JsonScanner {
public:
    explicit JsonScanner(const std::string& s) : s_(s) {}

    void skipWs() {
        while (p_ < s_.size()) {
            const char c = s_[p_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++p_;
                continue;
            }
            // Be tolerant of single-line // comments and /* ... */ blocks
            // even though strict JSON disallows them — the S0 fitter
            // never emits comments today, but this keeps the reader
            // forgiving for hand-edited debug copies.
            if (c == '/' && p_ + 1 < s_.size() && s_[p_+1] == '/') {
                while (p_ < s_.size() && s_[p_] != '\n') ++p_;
                continue;
            }
            if (c == '/' && p_ + 1 < s_.size() && s_[p_+1] == '*') {
                p_ += 2;
                while (p_ + 1 < s_.size() && !(s_[p_] == '*' && s_[p_+1] == '/')) ++p_;
                if (p_ + 1 < s_.size()) p_ += 2;
                continue;
            }
            break;
        }
    }

    char peek() {
        skipWs();
        return (p_ < s_.size()) ? s_[p_] : '\0';
    }

    char get() {
        skipWs();
        if (p_ >= s_.size()) throw parseError("unexpected end of input");
        return s_[p_++];
    }

    void expect(char c) {
        const char got = get();
        if (got != c) {
            std::ostringstream oss;
            oss << "expected '" << c << "', got '" << got << "'";
            throw parseError(oss.str());
        }
    }

    bool match(char c) {
        if (peek() == c) { ++p_; return true; }
        return false;
    }

    /// Read a JSON string (quoted). Supports \" \\ \/ \b \f \n \r \t and
    /// \uXXXX escapes; the escapes are not exercised by the S0 fitter
    /// output but the implementation keeps the reader robust.
    std::string readString() {
        skipWs();
        if (p_ >= s_.size() || s_[p_] != '"') throw parseError("expected string");
        ++p_;
        std::string out;
        while (p_ < s_.size() && s_[p_] != '"') {
            char c = s_[p_++];
            if (c == '\\') {
                if (p_ >= s_.size()) throw parseError("unterminated escape");
                char esc = s_[p_++];
                switch (esc) {
                    case '"':  out.push_back('"');  break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/');  break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    case 'u': {
                        // Skip \uXXXX (we don't expect non-ASCII rank keys).
                        if (p_ + 4 > s_.size()) throw parseError("bad \\u escape");
                        p_ += 4;
                        out.push_back('?');
                        break;
                    }
                    default: throw parseError("unknown escape");
                }
            } else {
                out.push_back(c);
            }
        }
        if (p_ >= s_.size()) throw parseError("unterminated string");
        ++p_; // consume closing "
        return out;
    }

    /// Read a JSON number (int or floating-point, with optional exponent).
    /// Uses std::strtod for FP-precision-correct conversion.
    double readNumber() {
        skipWs();
        const char* start = s_.c_str() + p_;
        char* end = nullptr;
        const double d = std::strtod(start, &end);
        if (end == start) throw parseError("expected number");
        p_ += static_cast<size_t>(end - start);
        return d;
    }

    /// Skip the value at the current position (object, array, string,
    /// number, bool, null) without retaining it.
    void skipValue() {
        const char c = peek();
        if (c == '{')      skipObject();
        else if (c == '[') skipArray();
        else if (c == '"') readString();
        else if (c == 't' || c == 'f') skipLiteral();
        else if (c == 'n') skipLiteral();
        else                readNumber();
    }

    /// Read a 1D array of numbers: `[n1, n2, ...]`.
    std::vector<double> readNumberArray1D() {
        std::vector<double> out;
        expect('[');
        if (match(']')) return out;
        while (true) {
            out.push_back(readNumber());
            if (match(',')) continue;
            expect(']');
            return out;
        }
    }

    /// Read a 2D array of numbers: `[[..], [..], ...]`.
    std::vector<std::vector<double>> readNumberArray2D() {
        std::vector<std::vector<double>> out;
        expect('[');
        if (match(']')) return out;
        while (true) {
            out.push_back(readNumberArray1D());
            if (match(',')) continue;
            expect(']');
            return out;
        }
    }

    /// Read a 3D array of numbers: `[[[..], [..]], ...]`.
    std::vector<std::vector<std::vector<double>>> readNumberArray3D() {
        std::vector<std::vector<std::vector<double>>> out;
        expect('[');
        if (match(']')) return out;
        while (true) {
            out.push_back(readNumberArray2D());
            if (match(',')) continue;
            expect(']');
            return out;
        }
    }

    bool atEnd() {
        skipWs();
        return p_ >= s_.size();
    }

private:
    std::runtime_error parseError(const std::string& msg) const {
        std::ostringstream oss;
        oss << "LeafShapeDistribution JSON parse error at offset " << p_
            << ": " << msg;
        return std::runtime_error(oss.str());
    }

    void skipObject() {
        expect('{');
        if (match('}')) return;
        while (true) {
            readString();
            expect(':');
            skipValue();
            if (match(',')) continue;
            expect('}');
            return;
        }
    }

    void skipArray() {
        expect('[');
        if (match(']')) return;
        while (true) {
            skipValue();
            if (match(',')) continue;
            expect(']');
            return;
        }
    }

    void skipLiteral() {
        // true / false / null — read identifier characters and discard.
        while (p_ < s_.size() && (std::isalpha(static_cast<unsigned char>(s_[p_])))) {
            ++p_;
        }
    }

    const std::string& s_;
    size_t p_ = 0;
};

// Read a key-keyed dict whose values are 1D number arrays (e.g. intercepts).
// Returns the values indexed by the integer parse of each key.
std::map<int, std::vector<double>> readIntKeyedNumberArray1DMap(JsonScanner& sc)
{
    std::map<int, std::vector<double>> out;
    sc.expect('{');
    if (sc.match('}')) return out;
    while (true) {
        const std::string key = sc.readString();
        sc.expect(':');
        const int k = std::atoi(key.c_str());
        out.emplace(k, sc.readNumberArray1D());
        if (sc.match(',')) continue;
        sc.expect('}');
        return out;
    }
}

// Read a key-keyed dict whose values are 3D arrays (asym residual grids
// shaped (n_u, n_v, 3) in the JSON, flattened to vector<Vector3d> here).
std::map<int, std::vector<Vector3d>> readIntKeyedAsymResidualMap(
    JsonScanner& sc, int n_u, int n_v)
{
    std::map<int, std::vector<Vector3d>> out;
    sc.expect('{');
    if (sc.match('}')) return out;
    while (true) {
        const std::string key = sc.readString();
        sc.expect(':');
        const int k = std::atoi(key.c_str());
        auto grid3d = sc.readNumberArray3D();
        if (static_cast<int>(grid3d.size()) != n_u) {
            std::ostringstream oss;
            oss << "asym_residual_grids_cm[" << k << "] has " << grid3d.size()
                << " u rows; expected n_u=" << n_u;
            throw std::invalid_argument(oss.str());
        }
        std::vector<Vector3d> flat;
        flat.reserve(static_cast<size_t>(n_u) * static_cast<size_t>(n_v));
        for (int iu = 0; iu < n_u; ++iu) {
            const auto& row = grid3d[iu];
            if (static_cast<int>(row.size()) != n_v) {
                std::ostringstream oss;
                oss << "asym_residual_grids_cm[" << k << "][" << iu << "] has "
                    << row.size() << " v entries; expected n_v=" << n_v;
                throw std::invalid_argument(oss.str());
            }
            for (int iv = 0; iv < n_v; ++iv) {
                const auto& xyz = row[iv];
                if (xyz.size() != 3) {
                    std::ostringstream oss;
                    oss << "asym_residual_grids_cm[" << k << "][" << iu << "][" << iv
                        << "] has " << xyz.size() << " components; expected 3";
                    throw std::invalid_argument(oss.str());
                }
                flat.emplace_back(xyz[0], xyz[1], xyz[2]);
            }
        }
        out.emplace(k, std::move(flat));
        if (sc.match(',')) continue;
        sc.expect('}');
        return out;
    }
}

// Read coeffs_block_layout: { "droop":[0,11], "along":[11,22], "halfwidth_norm":[22,33] }
struct CoeffsBlockLayout {
    int droop_start = 0;
    int along_start = 0;
    int halfwidth_start = 0;
    int n_cp = 0;
};

CoeffsBlockLayout readCoeffsBlockLayout(JsonScanner& sc)
{
    CoeffsBlockLayout out;
    sc.expect('{');
    if (sc.match('}')) {
        throw std::invalid_argument("coeffs_block_layout: empty object");
    }
    while (true) {
        const std::string key = sc.readString();
        sc.expect(':');
        const auto range = sc.readNumberArray1D();
        if (range.size() != 2) {
            throw std::invalid_argument("coeffs_block_layout: each entry must be [start, end]");
        }
        const int start = static_cast<int>(range[0]);
        const int end_ = static_cast<int>(range[1]);
        const int len = end_ - start;
        if (key == "droop") {
            out.droop_start = start;
            if (out.n_cp == 0) out.n_cp = len;
            else if (out.n_cp != len) {
                throw std::invalid_argument("coeffs_block_layout: block lengths differ");
            }
        } else if (key == "along") {
            out.along_start = start;
            if (out.n_cp == 0) out.n_cp = len;
            else if (out.n_cp != len) {
                throw std::invalid_argument("coeffs_block_layout: block lengths differ");
            }
        } else if (key == "halfwidth_norm") {
            out.halfwidth_start = start;
            if (out.n_cp == 0) out.n_cp = len;
            else if (out.n_cp != len) {
                throw std::invalid_argument("coeffs_block_layout: block lengths differ");
            }
        }
        if (sc.match(',')) continue;
        sc.expect('}');
        return out;
    }
}

// Process-wide cache; weak_ptr lets the distribution unload once every
// LeafRandomParameter holding a reference goes out of scope.
std::map<std::string, std::weak_ptr<LeafShapeDistribution>>& cache()
{
    static std::map<std::string, std::weak_ptr<LeafShapeDistribution>> c;
    return c;
}

std::mutex& cache_mutex()
{
    static std::mutex m;
    return m;
}

} // anonymous

// ============================================================
// LeafShapeDistribution::load — JSON file I/O + cache lookup
// ============================================================

std::shared_ptr<LeafShapeDistribution> LeafShapeDistribution::load(const std::string& path)
{
    {
        std::lock_guard<std::mutex> lk(cache_mutex());
        auto it = cache().find(path);
        if (it != cache().end()) {
            if (auto sp = it->second.lock()) return sp;
        }
    }

    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        std::ostringstream oss;
        oss << "LeafShapeDistribution::load: cannot open '" << path << "'. "
            << "Set shape_distribution_path to a valid file or leave it empty "
            << "to fall back to MedianLeafShape (no parametric variation).";
        throw std::runtime_error(oss.str());
    }
    std::ostringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();

    JsonScanner sc(text);
    sc.expect('{');

    auto out = std::shared_ptr<LeafShapeDistribution>(new LeafShapeDistribution());
    out->path_ = path;

    std::map<int, std::vector<double>> intercepts;
    std::map<int, std::vector<Vector3d>> asym_residuals;
    CoeffsBlockLayout layout;
    bool has_layout = false;

    if (!sc.match('}')) {
        while (true) {
            const std::string key = sc.readString();
            sc.expect(':');
            if (key == "n_components") {
                out->n_components_ = static_cast<int>(sc.readNumber());
            } else if (key == "n_cp_per_axis") {
                out->n_cp_per_axis_ = static_cast<int>(sc.readNumber());
            } else if (key == "spline_degree") {
                out->spline_degree_ = static_cast<int>(sc.readNumber());
            } else if (key == "spline_knots_u") {
                out->spline_knots_u_ = sc.readNumberArray1D();
            } else if (key == "n_u") {
                out->n_u_ = static_cast<int>(sc.readNumber());
            } else if (key == "n_v") {
                out->n_v_ = static_cast<int>(sc.readNumber());
            } else if (key == "n_ranks") {
                out->n_ranks_ = static_cast<int>(sc.readNumber());
            } else if (key == "intercepts") {
                intercepts = readIntKeyedNumberArray1DMap(sc);
            } else if (key == "asym_residual_grids_cm") {
                if (out->n_u_ < 2 || out->n_v_ < 2) {
                    throw std::invalid_argument(
                        "LeafShapeDistribution: 'n_u' and 'n_v' must appear "
                        "before 'asym_residual_grids_cm' in the JSON");
                }
                asym_residuals = readIntKeyedAsymResidualMap(sc, out->n_u_, out->n_v_);
            } else if (key == "cholesky_factor") {
                out->cholesky_factor_ = sc.readNumberArray2D();
            } else if (key == "coeffs_block_layout") {
                layout = readCoeffsBlockLayout(sc);
                has_layout = true;
            } else {
                sc.skipValue();
            }
            if (sc.match(',')) continue;
            sc.expect('}');
            break;
        }
    }

    // Validate.
    if (out->n_components_ <= 0 || out->n_cp_per_axis_ <= 0
        || out->spline_degree_ <= 0 || out->n_ranks_ <= 0
        || out->n_u_ < 2 || out->n_v_ < 2) {
        throw std::invalid_argument(
            "LeafShapeDistribution: required scalar fields missing or non-positive");
    }
    const int knots_expected = out->n_cp_per_axis_ + out->spline_degree_ + 1;
    if (static_cast<int>(out->spline_knots_u_.size()) != knots_expected) {
        std::ostringstream oss;
        oss << "LeafShapeDistribution: spline_knots_u length "
            << out->spline_knots_u_.size() << " != n_cp_per_axis + spline_degree + 1 = "
            << knots_expected;
        throw std::invalid_argument(oss.str());
    }
    if (static_cast<int>(intercepts.size()) != out->n_ranks_) {
        std::ostringstream oss;
        oss << "LeafShapeDistribution: intercepts has " << intercepts.size()
            << " entries; expected n_ranks=" << out->n_ranks_;
        throw std::invalid_argument(oss.str());
    }
    if (static_cast<int>(asym_residuals.size()) != out->n_ranks_) {
        std::ostringstream oss;
        oss << "LeafShapeDistribution: asym_residual_grids_cm has " << asym_residuals.size()
            << " entries; expected n_ranks=" << out->n_ranks_;
        throw std::invalid_argument(oss.str());
    }
    if (static_cast<int>(out->cholesky_factor_.size()) != out->n_components_) {
        std::ostringstream oss;
        oss << "LeafShapeDistribution: cholesky_factor has "
            << out->cholesky_factor_.size() << " rows; expected "
            << out->n_components_;
        throw std::invalid_argument(oss.str());
    }
    for (const auto& row : out->cholesky_factor_) {
        if (static_cast<int>(row.size()) != out->n_components_) {
            throw std::invalid_argument(
                "LeafShapeDistribution: cholesky_factor row width != n_components");
        }
    }

    // Repack intercepts and residuals into rank-indexed contiguous vectors.
    out->intercepts_.assign(out->n_ranks_, std::vector<double>());
    out->asym_residuals_.assign(out->n_ranks_, std::vector<Vector3d>());
    for (int r = 0; r < out->n_ranks_; ++r) {
        auto it = intercepts.find(r);
        if (it == intercepts.end()) {
            std::ostringstream oss;
            oss << "LeafShapeDistribution: intercepts['" << r << "'] missing";
            throw std::invalid_argument(oss.str());
        }
        if (static_cast<int>(it->second.size()) != out->n_components_) {
            std::ostringstream oss;
            oss << "LeafShapeDistribution: intercepts['" << r << "'] length "
                << it->second.size() << " != n_components=" << out->n_components_;
            throw std::invalid_argument(oss.str());
        }
        out->intercepts_[r] = std::move(it->second);

        auto ait = asym_residuals.find(r);
        if (ait == asym_residuals.end()) {
            std::ostringstream oss;
            oss << "LeafShapeDistribution: asym_residual_grids_cm['" << r << "'] missing";
            throw std::invalid_argument(oss.str());
        }
        out->asym_residuals_[r] = std::move(ait->second);
    }

    if (has_layout) {
        if (layout.n_cp != out->n_cp_per_axis_) {
            throw std::invalid_argument(
                "LeafShapeDistribution: coeffs_block_layout block length != n_cp_per_axis");
        }
        out->droop_block_start_     = layout.droop_start;
        out->along_block_start_     = layout.along_start;
        out->halfwidth_block_start_ = layout.halfwidth_start;
    } else {
        // Fall back to the canonical S0 layout (droop|along|halfwidth, each n_cp).
        out->droop_block_start_     = 0;
        out->along_block_start_     = out->n_cp_per_axis_;
        out->halfwidth_block_start_ = 2 * out->n_cp_per_axis_;
    }

    {
        std::lock_guard<std::mutex> lk(cache_mutex());
        cache()[path] = std::weak_ptr<LeafShapeDistribution>(out);
    }
    return out;
}

// ============================================================
// LeafShapeDistribution::makeShape — per-plant draw + ParametricLeafShape
// ============================================================

std::shared_ptr<ParametricLeafShape> LeafShapeDistribution::makeShape(
    int rank, double scale, unsigned int plant_seed_val) const
{
    if (rank < 0 || rank >= n_ranks_) {
        std::ostringstream oss;
        oss << "LeafShapeDistribution::makeShape: rank " << rank
            << " out of range [0, " << n_ranks_ << ")";
        throw std::out_of_range(oss.str());
    }

    const auto& base = intercepts_[rank];
    std::vector<double> coeffs = base;

    if (scale != 0.0) {
        // Local RNG so the Organism's master std::mt19937 stream is not
        // perturbed (D.0 6-XML invariance is preserved on the master
        // stream regardless of the shape draw). Same plant_seed_val →
        // same z across all ranks of this plant (D2 per-plant coherence).
        std::mt19937 rng(plant_seed_val ^ shape_seed_salt_);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::vector<double> z(n_components_);
        for (int i = 0; i < n_components_; ++i) z[i] = nd(rng);

        // delta = scale * (L @ z); L is lower-triangular.
        for (int i = 0; i < n_components_; ++i) {
            double sum = 0.0;
            const auto& row = cholesky_factor_[i];
            for (int j = 0; j <= i; ++j) {
                sum += row[j] * z[j];
            }
            coeffs[i] += scale * sum;
        }
    }

    std::vector<double> droop(coeffs.begin() + droop_block_start_,
                              coeffs.begin() + droop_block_start_ + n_cp_per_axis_);
    std::vector<double> along(coeffs.begin() + along_block_start_,
                              coeffs.begin() + along_block_start_ + n_cp_per_axis_);
    std::vector<double> halfwidth(coeffs.begin() + halfwidth_block_start_,
                                  coeffs.begin() + halfwidth_block_start_ + n_cp_per_axis_);

    return std::make_shared<ParametricLeafShape>(
        rank,
        spline_knots_u_,
        spline_degree_,
        std::move(droop),
        std::move(along),
        std::move(halfwidth),
        asym_residuals_[rank],   // copy — frozen across plants
        n_u_, n_v_);
}

} // namespace CPlantBox
