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
    std::map<int, double> max_w_xml;   ///< per-rank peak half-width (cm); S6 max_w bake
    std::map<int, double> lmax_intercept;  ///< per-rank fit-time midrib arc length (cm); fix 2b
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
            } else if (key == "max_w_xml_cm") {
                // S6 max_w bake: per-rank fit-time peak half-width (cm).
                // JSON shape: { "0": 2.57, "1": 3.43, ..., "14": 4.85 } —
                // string-keyed dict of scalars (not arrays). Inline parser
                // mirrors readIntKeyedNumberArray1DMap but reads readNumber()
                // instead of readNumberArray1D().
                sc.expect('{');
                if (!sc.match('}')) {
                    while (true) {
                        const std::string key2 = sc.readString();
                        sc.expect(':');
                        const int k = std::atoi(key2.c_str());
                        max_w_xml.emplace(k, sc.readNumber());
                        if (sc.match(',')) continue;
                        sc.expect('}');
                        break;
                    }
                }
            } else if (key == "lmax_intercept_cm") {
                // Fix 2b: per-rank fit-time midrib arc length (cm) — the
                // divisor the fitter used to normalise droop + along into
                // dimensionless shape coefficients. The C++ evaluator
                // multiplies the splines by this scalar to recover absolute
                // cm. Same JSON shape as max_w_xml_cm: {"0": 50.1, ...}.
                sc.expect('{');
                if (!sc.match('}')) {
                    while (true) {
                        const std::string key2 = sc.readString();
                        sc.expect(':');
                        const int k = std::atoi(key2.c_str());
                        lmax_intercept.emplace(k, sc.readNumber());
                        if (sc.match(',')) continue;
                        sc.expect('}');
                        break;
                    }
                }
            } else if (key == "coeffs_block_layout") {
                layout = readCoeffsBlockLayout(sc);
                has_layout = true;
            } else if (key == "pca_truncation") {
                // Fix path α (PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1):
                // top-K eigendecomposition of the pooled covariance Σ.
                // ``null`` → fall back to cholesky_factor (legacy path).
                // Otherwise ``{K, pca_components, pca_eigenvalues, ...}`` —
                // ``pca_components`` is K rows × n_components (each row =
                // eigenvector), ``pca_eigenvalues`` length K. We parse only
                // the fields we consume; auxiliary metadata
                // (retained_variance_fraction, all_eigenvalues_descending,
                // ...) is skipped via skipValue().
                if (sc.match('n')) {
                    // null literal — three more chars: 'u', 'l', 'l'.
                    sc.expect('u'); sc.expect('l'); sc.expect('l');
                    out->pca_K_ = 0;
                } else {
                    sc.expect('{');
                    if (!sc.match('}')) {
                        while (true) {
                            const std::string subkey = sc.readString();
                            sc.expect(':');
                            if (subkey == "K" || subkey == "n_components") {
                                out->pca_K_ = static_cast<int>(sc.readNumber());
                            } else if (subkey == "pca_components") {
                                out->pca_components_ = sc.readNumberArray2D();
                            } else if (subkey == "pca_eigenvalues") {
                                out->pca_eigenvalues_ = sc.readNumberArray1D();
                            } else {
                                sc.skipValue();
                            }
                            if (sc.match(',')) continue;
                            sc.expect('}');
                            break;
                        }
                    }
                }
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
    if (static_cast<int>(max_w_xml.size()) != out->n_ranks_) {
        std::ostringstream oss;
        oss << "LeafShapeDistribution: max_w_xml_cm has " << max_w_xml.size()
            << " entries; expected n_ranks=" << out->n_ranks_;
        throw std::invalid_argument(oss.str());
    }
    if (static_cast<int>(lmax_intercept.size()) != out->n_ranks_) {
        std::ostringstream oss;
        oss << "LeafShapeDistribution: lmax_intercept_cm has " << lmax_intercept.size()
            << " entries; expected n_ranks=" << out->n_ranks_
            << " (fix 2b — refit JSON via fit_parametric_leaf_shape.py)";
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

    // PCA truncation validation (fix path α). pca_K_ == 0 → legacy Cholesky
    // fallback (no validation needed). pca_K_ > 0 → both arrays must be
    // shape-correct; otherwise fail loudly because the runtime would
    // silently produce wrong shapes.
    if (out->pca_K_ > 0) {
        if (out->pca_K_ > out->n_components_) {
            std::ostringstream oss;
            oss << "LeafShapeDistribution: pca_K=" << out->pca_K_
                << " > n_components=" << out->n_components_;
            throw std::invalid_argument(oss.str());
        }
        if (static_cast<int>(out->pca_components_.size()) != out->pca_K_) {
            std::ostringstream oss;
            oss << "LeafShapeDistribution: pca_components has "
                << out->pca_components_.size() << " rows; expected pca_K="
                << out->pca_K_;
            throw std::invalid_argument(oss.str());
        }
        for (const auto& row : out->pca_components_) {
            if (static_cast<int>(row.size()) != out->n_components_) {
                throw std::invalid_argument(
                    "LeafShapeDistribution: pca_components row width != n_components");
            }
        }
        if (static_cast<int>(out->pca_eigenvalues_.size()) != out->pca_K_) {
            std::ostringstream oss;
            oss << "LeafShapeDistribution: pca_eigenvalues has "
                << out->pca_eigenvalues_.size() << " entries; expected pca_K="
                << out->pca_K_;
            throw std::invalid_argument(oss.str());
        }
    }

    // Repack intercepts and residuals into rank-indexed contiguous vectors.
    out->intercepts_.assign(out->n_ranks_, std::vector<double>());
    out->asym_residuals_.assign(out->n_ranks_, std::vector<Vector3d>());
    out->max_w_per_rank_.assign(out->n_ranks_, 0.0);
    out->lmax_per_rank_.assign(out->n_ranks_, 0.0);
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

        auto mit = max_w_xml.find(r);
        if (mit == max_w_xml.end()) {
            std::ostringstream oss;
            oss << "LeafShapeDistribution: max_w_xml_cm['" << r << "'] missing";
            throw std::invalid_argument(oss.str());
        }
        if (!(mit->second > 0.0)) {
            std::ostringstream oss;
            oss << "LeafShapeDistribution: max_w_xml_cm['" << r << "'] = "
                << mit->second << "; expected > 0";
            throw std::invalid_argument(oss.str());
        }
        out->max_w_per_rank_[r] = mit->second;

        auto lit = lmax_intercept.find(r);
        if (lit == lmax_intercept.end()) {
            std::ostringstream oss;
            oss << "LeafShapeDistribution: lmax_intercept_cm['" << r << "'] missing";
            throw std::invalid_argument(oss.str());
        }
        if (!(lit->second > 0.0)) {
            std::ostringstream oss;
            oss << "LeafShapeDistribution: lmax_intercept_cm['" << r << "'] = "
                << lit->second << "; expected > 0";
            throw std::invalid_argument(oss.str());
        }
        out->lmax_per_rank_[r] = lit->second;
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

        if (pca_K_ > 0) {
            // Fix path α: PCA-truncated draw. ``pca_components_`` is K rows
            // × n_components_ (each row = one eigenvector of Σ);
            // ``pca_eigenvalues_`` is the length-K vector of corresponding
            // eigenvalues. Sample z_K ~ N(0, I_K) and reconstruct
            // ``delta = U_K · √Λ_K · z_K``: scalar per-mode amplitudes
            // (`sqrt(λ_k) · z_k`) projected back through the eigenbasis.
            // Drops the noise modes (K dropped of N=33) responsible for
            // root cause #2's negative halfwidth / oscillating droop /
            // non-monotonic along.
            std::vector<double> z_K(pca_K_);
            for (int k = 0; k < pca_K_; ++k) z_K[k] = nd(rng);
            for (int k = 0; k < pca_K_; ++k) {
                const double w = scale * std::sqrt(std::max(pca_eigenvalues_[k], 0.0)) * z_K[k];
                const auto& vec = pca_components_[k];
                for (int i = 0; i < n_components_; ++i) {
                    coeffs[i] += w * vec[i];
                }
            }
        } else {
            // Legacy fallback: full Cholesky draw.
            // delta = scale * (L @ z); L is lower-triangular.
            std::vector<double> z(n_components_);
            for (int i = 0; i < n_components_; ++i) z[i] = nd(rng);
            for (int i = 0; i < n_components_; ++i) {
                double sum = 0.0;
                const auto& row = cholesky_factor_[i];
                for (int j = 0; j <= i; ++j) {
                    sum += row[j] * z[j];
                }
                coeffs[i] += scale * sum;
            }
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
        n_u_, n_v_,
        max_w_per_rank_[rank],   // S6 max_w bake — fit-time peak half-width
        lmax_per_rank_[rank]);   // fix 2b — fit-time midrib arc length
}

} // namespace CPlantBox
