#ifndef ALGOIM_QUADRATURE_MULTIPOLY_HPP
#define ALGOIM_QUADRATURE_MULTIPOLY_HPP

#define ALGOIM_M 8

#include "/home/trobbiani/Codici/long-term-encounters/LAAS_CNRS/MyProfiler.hpp"
#include "bernstein.hpp"
#include "booluarray.hpp"
#include "gaussquad.hpp"
#include "multiloop.hpp"
#include "polyset.hpp"
#include "real.hpp"
#include "sparkstack.hpp"
#include "tanhsinh.hpp"
#include "uvector.hpp"
#include "xarray.hpp"
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace algoim {
namespace detail {
template <int N>
booluarray<N, ALGOIM_M>
mask_driver(const xarray<real, N> &f, const booluarray<N, ALGOIM_M> &fmask,
            const xarray<real, N> *g, const booluarray<N, ALGOIM_M> *gmask) {
  PROFILE_SCOPE("algoim::mask_driver");
  booluarray<N, ALGOIM_M> mask(false);
  auto helper = [&](auto &&self, uvector<int, N> a, uvector<int, N> b) {
    bool overlap = false;
    for (MultiLoop<N> i(a, b); ~i; ++i)
      if (fmask(i()) && (!gmask || (*gmask)(i())))
        overlap = true;
    if (!overlap)
      return;

    real eps = 0.015625 / ALGOIM_M;
    uvector<real, N> xa, xb;
    for (int dim = 0; dim < N; ++dim) {
      xa(dim) = real(a(dim)) / ALGOIM_M - eps;
      xb(dim) = real(b(dim)) / ALGOIM_M + eps;
    }

    if (g) {
      xarray<real, N> fab(nullptr, f.ext()), gab(nullptr, g->ext());
      algoim_spark_alloc(real, fab, gab);
      bernstein::deCasteljau(f, xa, xb, fab);
      bernstein::deCasteljau(*g, xa, xb, gab);
      if (bernstein::orthantTest(fab, gab))
        return;
    } else {
      xarray<real, N> fab(nullptr, f.ext());
      algoim_spark_alloc(real, fab);
      bernstein::deCasteljau(f, xa, xb, fab);
      if (bernstein::uniformSign(fab) != 0)
        return;
    }

    if (b(0) - a(0) == 1) {
      assert(all(b - a == 1));
      assert(fmask(a) && (!gmask || (*gmask)(a)));
      mask(a) = true;
      return;
    }

    assert(all(b - a > 1) && all((b - a) % 2 == 0));
    uvector<int, N> delta = (b - a) / 2;
    for (MultiLoop<N> i(0, 2); ~i; ++i)
      self(self, a + i() * delta, a + (i() + 1) * delta);
  };
  helper(helper, 0, ALGOIM_M);
  return mask;
}

template <int N>
booluarray<N, ALGOIM_M> intersectionMask(const xarray<real, N> &f,
                                         const booluarray<N, ALGOIM_M> &fmask,
                                         const xarray<real, N> &g,
                                         const booluarray<N, ALGOIM_M> &gmask) {
  return mask_driver(f, fmask, &g, &gmask);
}

template <int N>
booluarray<N, ALGOIM_M> nonzeroMask(const xarray<real, N> &f,
                                    const booluarray<N, ALGOIM_M> &fmask) {
  return mask_driver<N>(f, fmask, nullptr, nullptr);
}

template <int N>
booluarray<N - 1, ALGOIM_M> collapseMask(const booluarray<N, ALGOIM_M> &mask,
                                         int k) {
  booluarray<N - 1, ALGOIM_M> r(false);
  for (MultiLoop<N> i(0, ALGOIM_M); ~i; ++i)
    if (mask(i()))
      r(remove_component(i(), k)) = true;
  return r;
}

template <int N> bool maskEmpty(const booluarray<N, ALGOIM_M> &mask) {
  return mask.none();
}

template <int N>
bool pointWithinMask(const booluarray<N, ALGOIM_M> &mask,
                     const uvector<real, N> &x) {
  using std::floor;
  uvector<int, N> cell;
  for (int dim = 0; dim < N; ++dim)
    cell(dim) = std::max(
        0, std::min(ALGOIM_M - 1, static_cast<int>(floor(x(dim) * ALGOIM_M))));
  return mask(cell);
}

template <int N>
bool lineIntersectsMask(const booluarray<N, ALGOIM_M> &mask,
                        const uvector<real, N - 1> &x, int k) {
  using std::floor;
  if constexpr (N > 1) {
    uvector<int, N> cell;
    for (int dim = 0; dim < N; ++dim)
      if (dim < k)
        cell(dim) =
            std::max(0, std::min(ALGOIM_M - 1,
                                 static_cast<int>(floor(x(dim) * ALGOIM_M))));
      else if (dim > k)
        cell(dim) = std::max(
            0, std::min(ALGOIM_M - 1,
                        static_cast<int>(floor(x(dim - 1) * ALGOIM_M))));
    for (int i = 0; i < ALGOIM_M; ++i) {
      cell(k) = i;
      if (mask(cell))
        return true;
    }
    return false;
  } else
    return !maskEmpty(mask);
}

template <int N>
void restrictToFace(const xarray<real, N> &a, int k, int side,
                    xarray<real, N - 1> &out) {
  assert(0 <= k && k < N && (side == 0 || side == 1));
  assert(all(out.ext() == remove_component(a.ext(), k)));
  int P = a.ext(k);
  for (auto i = out.loop(); ~i; ++i) {
    uvector<int, N> j;
    for (int dim = 0; dim < N; ++dim)
      j(dim) = (dim < k) ? i(dim) : ((dim == k) ? side * (P - 1) : i(dim - 1));
    out.l(i) = a.m(j);
  }
}

template <int N>
booluarray<N - 1, ALGOIM_M> restrictToFace(const booluarray<N, ALGOIM_M> &a,
                                           int k, int side) {
  assert(0 <= k && k < N && (side == 0 || side == 1));
  booluarray<N - 1, ALGOIM_M> r;
  for (MultiLoop<N - 1> i(0, ALGOIM_M); ~i; ++i) {
    uvector<int, N> j;
    for (int dim = 0; dim < N; ++dim)
      j(dim) = (dim < k) ? i(dim)
                         : ((dim == k) ? side * (ALGOIM_M - 1) : i(dim - 1));
    r(i()) = a(j);
  }
  return r;
}

template <typename T> T det_qr(xarray<T, 2> &A, int &rank, T tol = 10.0) {
  PROFILE_SCOPE("algoim::det_qr");
  assert(A.ext(0) == A.ext(1) && A.ext(0) > 0);
  using std::abs;
  using std::max;
  T det = 1.0;
  int n = A.ext(0);
  T max_diag_r = 0.0;
  for (int j = 0; j < n; ++j) {
    T m = -1;
    int k = -1;
    for (int i = j; i < n; ++i) {
      T mag = 0;
      for (int a = 0; a < n; ++a)
        mag += util::sqr(A(a, i));
      if (k == -1 || mag >= m) {
        m = mag;
        k = i;
      }
    }
    assert(j <= k && k < n);
    if (k != j) {
      for (int a = 0; a < n; ++a)
        std::swap(A(a, j), A(a, k));
      det *= -1.0;
    }
    for (int i = n - 1; i >= j + 1; --i) {
      T c, s;
      util::givens_get(A(i - 1, j), A(i, j), c, s);
      for (int k = j; k < n; ++k)
        util::givens_rotate(A(i - 1, k), A(i, k), c, s);
    }
    det *= A(j, j);
    max_diag_r = max(max_diag_r, abs(A(j, j)));
  }

  tol *= max_diag_r * n * std::numeric_limits<T>::epsilon();
  rank = 0;
  for (int i = 0; i < n; ++i)
    if (abs(A(i, i)) > tol)
      ++rank;

  return det;
}

template <int N>
uvector<int, N - 1> resultantExtent(const uvector<int, N> &p,
                                    const uvector<int, N> &q, int dim) {
  uvector<int, N - 1> ext;
  for (int i = 0; i < N - 1; ++i) {
    int ii = (i < dim) ? i : i + 1;
    ext(i) = (p(dim) - 1) * (q(ii) - 1) + (q(dim) - 1) * (p(ii) - 1) + 1;
  }
  return ext;
}

template <int N>
uvector<int, N - 1> discriminantExtent(const uvector<int, N> &p, int dim) {
  uvector<int, N - 1> ext;
  for (int i = 0; i < N - 1; ++i) {
    int ii = (i < dim) ? i : i + 1;
    ext(i) = (2 * p(dim) - 3) * (p(ii) - 1) + 1;
  }
  return ext;
}

template <int N>
bool resultant_core(const xarray<real, N> &p, const xarray<real, N> *q, int k,
                    xarray<real, N - 1> &out) {
  PROFILE_SCOPE("algoim::resultant_core");
  assert(0 <= k && k < N);

  int P = p.ext(k);
  int Q = q ? q->ext(k) : P - 1;
  int M = (P == Q) ? P - 1 : P + Q - 2;
  assert(P >= 2 && Q >= 1 && M >= 1);

  xarray<real, N - 1> f(nullptr, out.ext());
  xarray<real, 2> mat(nullptr, uvector<int, 2>{M, M});
  real *pk, *qk;
  algoim_spark_alloc(real, f, mat);
  algoim_spark_alloc(real, &pk, P, &qk, Q);

  for (auto i = f.loop(); ~i; ++i) {
    uvector<real, N - 1> x;
    for (int dim = 0; dim < N - 1; ++dim)
      x(dim) = bernstein::modifiedChebyshevNode(i(dim), f.ext(dim));

    bernstein::collapseAlongAxis(p, x, k, pk);
    if (q)
      bernstein::collapseAlongAxis(*q, x, k, qk);
    else
      bernstein::bernsteinDerivative(pk, P, qk);

    if (P == Q)
      bernstein::bezoutMatrix(pk, qk, P, mat);
    else
      bernstein::sylvesterMatrix(pk, P, qk, Q, mat);

    int rank;
    f.l(i) = det_qr(mat, rank);
  }

  bernstein::normalise(f);
  bernstein::bernsteinInterpolate(
      f, std::pow(100.0 * std::numeric_limits<real>::epsilon(), 1.0 / (N - 1)),
      out);

  bool b =
      bernstein::autoReduction(out, 1e4 * std::numeric_limits<real>::epsilon());

  if (b)
    resultant_core(p, q, k, out);
  return true;
}

template <int N>
bool resultant(const xarray<real, N> &p, const xarray<real, N> &q, int k,
               xarray<real, N - 1> &out) {
  return resultant_core(p, &q, k, out);
}

template <int N>
bool discriminant(const xarray<real, N> &p, int k, xarray<real, N - 1> &out) {
  xarray<real, N> prime(nullptr, inc_component(p.ext(), k, -1));
  algoim_spark_alloc(real, prime);
  bernstein::bernsteinDerivative(p, k, prime);
  return resultant_core(p, &prime, k, out);
}

template <int N>
void eliminate_axis(PolySet<N, ALGOIM_M> &phi, int k,
                    PolySet<N - 1, ALGOIM_M> &psi) {
  PROFILE_SCOPE("algoim::eliminate_axis");
  static_assert(N >= 2, "N >= 2 required to eliminate axis");
  assert(0 <= k && k < N);
  assert(psi.count() == 0);

  for (size_t i = 0; i < static_cast<size_t>(phi.count()); ++i) {
    const auto &p = phi.poly(i);
    const auto &mask = phi.mask(i);

    for (int side = 0; side <= 1; ++side) {
      xarray<real, N - 1> p_face(nullptr, remove_component(p.ext(), k));
      algoim_spark_alloc(real, p_face);
      restrictToFace(p, k, side, p_face);
      auto p_face_mask = nonzeroMask(p_face, restrictToFace(mask, k, side));
      if (!maskEmpty(p_face_mask)) {
        bernstein::autoReduction(p_face);
        bernstein::normalise(p_face);
        psi.push_back(p_face, p_face_mask);
      }
    }

    xarray<real, N> p_k(nullptr, p.ext());
    algoim_spark_alloc(real, p_k);
    bernstein::elevatedDerivative(p, k, p_k);
    auto disc_mask = intersectionMask(p, mask, p_k, mask);
    if (!maskEmpty(disc_mask)) {
      uvector<int, N - 1> R = discriminantExtent(p.ext(), k);
      xarray<real, N - 1> disc(nullptr, R);
      algoim_spark_alloc(real, disc);
      if (discriminant(p, k, disc)) {
        bernstein::normalise(disc);
        psi.push_back(disc, collapseMask(disc_mask, k));
      }
    }
  }

  for (size_t i = 0; i < static_cast<size_t>(phi.count()); ++i)
    for (size_t j = i + 1; j < static_cast<size_t>(phi.count()); ++j) {
      const auto &p = phi.poly(i);
      const auto &pmask = phi.mask(i);
      const auto &q = phi.poly(j);
      const auto &qmask = phi.mask(j);
      auto mask = intersectionMask(p, pmask, q, qmask);
      if (!maskEmpty(mask)) {
        uvector<int, N - 1> R = resultantExtent(p.ext(), q.ext(), k);
        xarray<real, N - 1> res(nullptr, R);
        algoim_spark_alloc(real, res);
        if (resultant(p, q, k, res)) {
          bernstein::normalise(res);
          psi.push_back(res, collapseMask(mask, k));
        }
      }
    };
}

template <int N>
booluarray<N, ALGOIM_M> positiveSubcellMask(const xarray<real, N> &f) {
  booluarray<N, ALGOIM_M> mask(false);

  real root_min = *f.data();
  real root_max = *f.data();
  for (auto i = f.loop(); ~i; ++i) {
    real v = f.l(i);
    if (v < root_min)
      root_min = v;
    if (v > root_max)
      root_max = v;
  }
  if (root_max <= 0.0)
    return mask;
  if (root_min > 0.0) {
    for (MultiLoop<N> i(0, ALGOIM_M); ~i; ++i)
      mask(i()) = true;
    return mask;
  }

  auto helper = [&](auto &&self, uvector<int, N> a, uvector<int, N> b) -> void {
    uvector<real, N> xa, xb;
    for (int dim = 0; dim < N; ++dim) {
      xa(dim) = real(a(dim)) / ALGOIM_M;
      xb(dim) = real(b(dim)) / ALGOIM_M;
    }

    xarray<real, N> fab(nullptr, f.ext());
    algoim_spark_alloc(real, fab);
    bernstein::deCasteljau(f, xa, xb, fab);

    real min_v = *fab.data();
    real max_v = *fab.data();
    for (auto i = fab.loop(); ~i; ++i) {
      real v = fab.l(i);
      if (v < min_v)
        min_v = v;
      if (v > max_v)
        max_v = v;
    }

    if (max_v <= 0.0)
      return;

    if (min_v > 0.0) {
      for (MultiLoop<N> i(a, b); ~i; ++i)
        mask(i()) = true;
      return;
    }

    if (b(0) - a(0) == 1)
      return;

    uvector<int, N> delta = (b - a) / 2;
    for (MultiLoop<N> i(0, 2); ~i; ++i)
      self(self, a + i() * delta, a + (i() + 1) * delta);
  };

  uvector<int, N> root_a(0), root_b(ALGOIM_M);
  uvector<int, N> delta = (root_b - root_a) / 2;
  for (MultiLoop<N> i(0, 2); ~i; ++i)
    helper(helper, root_a + i() * delta, root_a + (i() + 1) * delta);

  return mask;
}

template <int N>
void eliminate_time_axis(PolySet<N, ALGOIM_M> &phi, int k,
                         const std::vector<bool> &skip_face_0,
                         const std::vector<bool> &skip_face_1,
                         PolySet<N - 1, ALGOIM_M> &psi) {
  PROFILE_SCOPE("algoim::eliminate_time_axis");
  static_assert(N >= 2, "N >= 2 required to eliminate axis");
  assert(0 <= k && k < N);
  assert(psi.count() == 0);

  booluarray<N - 1, ALGOIM_M> covered_mask(false);
  if (phi.count() > 1) {
    for (size_t i = 0; i < static_cast<size_t>(phi.count()); ++i) {
      const auto &p = phi.poly(i);
      uvector<int, N - 1> ext_3d = remove_component(p.ext(), k);

      for (int side = 0; side <= 1; ++side) {
        xarray<real, N - 1> p_face(nullptr, ext_3d);
        algoim_spark_alloc(real, p_face);
        restrictToFace(p, k, side, p_face);
        auto pos_mask = positiveSubcellMask<N - 1>(p_face);
        for (MultiLoop<N - 1> idx(0, ALGOIM_M); ~idx; ++idx)
          if (pos_mask(idx()))
            covered_mask(idx()) = true;
      }

      xarray<real, N - 1> p_mid(nullptr, ext_3d);
      real *pline;
      int P_t = p.ext(k);
      algoim_spark_alloc(real, p_mid);
      algoim_spark_alloc(real, &pline, P_t);
      const real *binom_row = Binomial::row(P_t - 1);

      for (auto idx = p_mid.loop(); ~idx; ++idx) {
        uvector<int, N> j_idx;
        for (int dim = 0; dim < N; ++dim)
          j_idx(dim) = (dim < k) ? idx(dim) : ((dim == k) ? 0 : idx(dim - 1));
        for (int t = 0; t < P_t; ++t) {
          j_idx(k) = t;
          pline[t] = p.m(j_idx);
        }
        real val = 0.0, deriv = 0.0;
        bernstein::bernsteinValueAndDerivative(pline, P_t, binom_row, real(0.5),
                                               val, deriv);
        p_mid.l(idx) = val;
      }
      auto pos_mid_mask = positiveSubcellMask<N - 1>(p_mid);
      for (MultiLoop<N - 1> idx(0, ALGOIM_M); ~idx; ++idx)
        if (pos_mid_mask(idx()))
          covered_mask(idx()) = true;
    }
  }

  for (size_t i = 0; i < static_cast<size_t>(phi.count()); ++i) {
    const auto &p = phi.poly(i);
    const auto &mask = phi.mask(i);

    for (int side = 0; side <= 1; ++side) {
      if (side == 0 && i < skip_face_0.size() && skip_face_0[i])
        continue;
      if (side == 1 && i < skip_face_1.size() && skip_face_1[i])
        continue;

      xarray<real, N - 1> p_face(nullptr, remove_component(p.ext(), k));
      algoim_spark_alloc(real, p_face);
      restrictToFace(p, k, side, p_face);

      auto base_face_mask = restrictToFace(mask, k, side);
      for (MultiLoop<N - 1> idx(0, ALGOIM_M); ~idx; ++idx)
        if (covered_mask(idx()))
          base_face_mask(idx()) = false;

      auto p_face_mask = nonzeroMask(p_face, base_face_mask);
      if (!maskEmpty(p_face_mask)) {
        bernstein::autoReduction(p_face);
        bernstein::normalise(p_face);
        psi.push_back(p_face, p_face_mask);
      }
    }

    xarray<real, N> p_k(nullptr, p.ext());
    algoim_spark_alloc(real, p_k);
    bernstein::elevatedDerivative(p, k, p_k);
    auto disc_mask = intersectionMask(p, mask, p_k, mask);
    if (!maskEmpty(disc_mask)) {
      auto disc_mask_3d = collapseMask(disc_mask, k);
      for (MultiLoop<N - 1> idx(0, ALGOIM_M); ~idx; ++idx)
        if (covered_mask(idx()))
          disc_mask_3d(idx()) = false;

      if (!maskEmpty(disc_mask_3d)) {
        uvector<int, N - 1> R = discriminantExtent(p.ext(), k);
        xarray<real, N - 1> disc(nullptr, R);
        algoim_spark_alloc(real, disc);
        if (discriminant(p, k, disc)) {
          bernstein::normalise(disc);
          psi.push_back(disc, disc_mask_3d);
        }
      }
    }
  }
}

template <int N>
uvector<real, N> score_estimate(PolySet<N, ALGOIM_M> &phi,
                                uvector<bool, N> &has_disc) {
  PROFILE_SCOPE("algoim::score_estimate");
  static_assert(N > 1, "score_estimate of practical use only with N > 1");
  using std::abs;
  uvector<real, N> s = 0;
  has_disc = false;
  for (size_t i = 0; i < static_cast<size_t>(phi.count()); ++i) {
    const auto &p = phi.poly(i);
    const auto &mask = phi.mask(i);

    for (MultiLoop<N> j(0, ALGOIM_M); ~j; ++j)
      if (mask(j())) {
        uvector<real, N> x = (j() + 0.5) / real(ALGOIM_M);
        uvector<real, N> g = bernstein::evalBernsteinPolyGradient(p, x);
        real sum = 0;
        for (int dim = 0; dim < N; ++dim) {
          g(dim) = abs(g(dim));
          sum += g(dim);
        }
        if (sum > 0)
          s += g / sum;
      }

    xarray<real, N> p_k(nullptr, p.ext());
    algoim_spark_alloc(real, p_k);
    for (int k = 0; k < N; ++k) {
      bernstein::elevatedDerivative(p, k, p_k);
      auto disc_mask = intersectionMask(p, mask, p_k, mask);
      has_disc(k) |= !maskEmpty(disc_mask);
    }
  }
  return s;
}
} // namespace detail

// --- STRUTTURE PER INTEGRAZIONE GAUSSIANA ---
template <int MaxDim = 6> struct GaussianQuadContext {
  double z_min[MaxDim];
  double z_max[MaxDim];
  double total_poc;
};

inline double gaussian_integral_1d(double z_a, double z_b) {
  constexpr double INV_SQRT_2 = 0.7071067811865475244;
  return 0.5 * (std::erf(z_b * INV_SQRT_2) - std::erf(z_a * INV_SQRT_2));
}

enum QuadStrategy { AlwaysGL, AlwaysTS, AutoMixed };

template <int N> struct ImplicitPolyQuadrature {
  enum IntegralType { Inner, OuterSingle, OuterAggregate };

  PolySet<N, ALGOIM_M> phi;
  int k;
  ImplicitPolyQuadrature<N - 1> base;
  bool auto_apply_TS;
  IntegralType type;
  std::array<std::tuple<int, ImplicitPolyQuadrature<N - 1>>, N - 1> base_other;

  ImplicitPolyQuadrature() : k(-1) {}

  ImplicitPolyQuadrature(const xarray<real, N> &p) {
    auto mask = detail::nonzeroMask(p, booluarray<N, ALGOIM_M>(true));
    if (!detail::maskEmpty(mask))
      phi.push_back(p, mask);
    build(true, false);
  }

  ImplicitPolyQuadrature(const xarray<real, N> &p, const xarray<real, N> &q) {
    {
      auto mask = detail::nonzeroMask(p, booluarray<N, ALGOIM_M>(true));
      if (!detail::maskEmpty(mask))
        phi.push_back(p, mask);
    }
    {
      auto mask = detail::nonzeroMask(q, booluarray<N, ALGOIM_M>(true));
      if (!detail::maskEmpty(mask))
        phi.push_back(q, mask);
    }
    build(true, false);
  }

  ImplicitPolyQuadrature(const xarray<real, N> &p,
                         const booluarray<N, ALGOIM_M> &pmask,
                         const xarray<real, N> &q,
                         const booluarray<N, ALGOIM_M> &qmask) {
    {
      auto mask = detail::nonzeroMask(p, pmask);
      if (!maskEmpty(mask))
        phi.push_back(p, mask);
    }
    {
      auto mask = detail::nonzeroMask(q, qmask);
      if (!maskEmpty(mask))
        phi.push_back(q, mask);
    }
    build(true, false);
  }

  void build(bool outer, bool auto_apply_TS) {
    PROFILE_SCOPE("algoim::ImplicitPolyQuad_build");
    type = outer ? OuterSingle : Inner;
    this->auto_apply_TS = auto_apply_TS;

    if (phi.count() == 0) {
      k = N;
      this->auto_apply_TS = false;
      return;
    }

    if constexpr (N == 1) {
      k = 0;
      return;
    } else {
      uvector<bool, N> has_disc;
      uvector<real, N> score = detail::score_estimate(phi, has_disc);
      assert(max(abs(score)) > 0);
      score /= 2 * max(abs(score));
      for (int i = 0; i < N; ++i)
        if (!has_disc(i))
          score(i) += 1.0;

      k = argmax(score);
      detail::eliminate_axis(phi, k, base.phi);
      base.build(false, this->auto_apply_TS || has_disc(k));

      if (outer && has_disc(k)) {
        type = OuterAggregate;
        for (int i = 0; i < N; ++i)
          if (i != k) {
            auto &[kother, base] = base_other[i < k ? i : i - 1];
            kother = i;
            detail::eliminate_axis(phi, kother, base.phi);
            base.build(false, this->auto_apply_TS || true);
          }
      }
    }
  }

  // --- INTEGRATORE ORIGINALE (lasciato intatto per evitare problemi sui
  // template) ---
  template <typename F> void integrate(QuadStrategy strategy, int q, F &&f) {
    assert(0 <= k && k <= N);

    if (k == N) {
      assert(!auto_apply_TS);
      for (MultiLoop<N> i(0, q); ~i; ++i) {
        uvector<real, N> x;
        real w = 1.0;
        for (int dim = 0; dim < N; ++dim) {
          x(dim) = GaussQuad::x(q, i(dim));
          w *= GaussQuad::w(q, i(dim));
        }
        f(x, w);
      }
      return;
    }

    int max_count = 2;
    for (size_t i = 0; i < static_cast<size_t>(phi.count()); ++i)
      max_count += phi.poly(i).ext(k) - 1;

    auto integrand = [&](const uvector<real, N - 1> &xbase, real w) {
      real *nodes;
      algoim_spark_alloc(real, &nodes, max_count);
      nodes[0] = 0.0;
      nodes[1] = 1.0;
      int count = 2;

      for (size_t i = 0; i < static_cast<size_t>(phi.count()); ++i) {
        const auto &p = phi.poly(i);
        const auto &mask = phi.mask(i);
        int P = p.ext(k);

        if (!detail::lineIntersectsMask(mask, xbase, k))
          continue;

        real *pline, *roots;
        algoim_spark_alloc(real, &pline, P, &roots, P - 1);
        bernstein::collapseAlongAxis(p, xbase, k, pline);
        int rcount = bernstein::bernsteinUnitIntervalRealRoots(pline, P, roots);

        for (int j = 0; j < rcount; ++j) {
          auto x = add_component(xbase, k, roots[j]);
          if (detail::pointWithinMask(mask, x))
            nodes[count++] = roots[j];
        }
      };

      std::sort(nodes, nodes + count);
      assert(nodes[0] == real(0) && nodes[count - 1] == real(1));

      real tol = 10.0 * std::numeric_limits<real>::epsilon();
      using std::abs;
      for (int i = 1; i < count - 1; ++i)
        if (abs(nodes[i]) < tol)
          nodes[i] = 0.0;
        else if (abs(nodes[i] - 1) < tol)
          nodes[i] = 1.0;
        else if (abs(nodes[i] - nodes[i + 1]) < tol)
          nodes[i + 1] = nodes[i];

      for (int i = 0; i < count - 1; ++i) {
        real x0 = nodes[i];
        real x1 = nodes[i + 1];
        if (x0 == x1)
          continue;

        bool GL = true;
        if (strategy == AlwaysTS)
          GL = false;
        if (strategy == AutoMixed)
          GL = !this->auto_apply_TS;

        if (GL)
          for (int j = 0; j < q; ++j)
            f(add_component(xbase, k, x0 + (x1 - x0) * GaussQuad::x(q, j)),
              w * (x1 - x0) * GaussQuad::w(q, j));
        else
          for (int j = 0; j < q; ++j)
            f(add_component(xbase, k, TanhSinhQuadrature::x(q, j, x0, x1)),
              w * TanhSinhQuadrature::w(q, j, x0, x1));
      }
    };

    if constexpr (N > 1)
      base.integrate(strategy, q, integrand);
    else
      integrand(uvector<real, 0>(), real(1));
  }

  // --- INTEGRATORE ORIGINALE SURF (lasciato intatto) ---
  template <typename F>
  void integrate_surf(QuadStrategy strategy, int q, F &&f) {
    static_assert(N > 1,
                  "surface integral only implemented in N > 1 dimensions");
    assert(type == OuterSingle || type == OuterAggregate);

    if (k == N)
      return;
    int k_active = -1;
    auto integrand = [&](const uvector<real, N - 1> &xbase, real w) {
      assert(0 <= k_active && k_active < N);
      for (size_t i = 0; i < static_cast<size_t>(phi.count()); ++i) {
        const auto &p = phi.poly(i);
        const auto &mask = phi.mask(i);
        int P = p.ext(k_active);

        if (!detail::lineIntersectsMask(mask, xbase, k_active))
          continue;

        real *pline, *roots;
        algoim_spark_alloc(real, &pline, P, &roots, P - 1);
        bernstein::collapseAlongAxis(p, xbase, k_active, pline);
        int rcount = bernstein::bernsteinUnitIntervalRealRoots(pline, P, roots);

        for (int j = 0; j < rcount; ++j) {
          auto x = add_component(xbase, k_active, roots[j]);
          if (detail::pointWithinMask(mask, x)) {
            using std::abs;
            uvector<real, N> g = bernstein::evalBernsteinPolyGradient(p, x);
            if (type == OuterAggregate) {
              real alpha = max(abs(g));
              if (alpha > 0) {
                g /= alpha;
                alpha = abs(g(k_active)) / norm(g);
              }
              f(x, w * alpha,
                set_component<real, N>(real(0.0), k_active,
                                       w * util::sign(g(k_active))));
            } else {
              uvector<real, N> n = g;
              if (norm(n) > 0)
                n *= real(1.0) / norm(n);
              real alpha = w * norm(g) / abs(g(k_active));
              f(x, alpha, alpha * n);
            }
          }
        }
      }
    };

    k_active = k;
    base.integrate(strategy, q, integrand);
    if (type == OuterAggregate) {
      for (int i = 0; i < N - 1; ++i) {
        auto &[k_oth, base_oth] = base_other[i];
        k_active = k_oth;
        base_oth.integrate(strategy, q, integrand);
      }
    }
  }

  // --- NUOVO METODO GAUSSIANO N-DIMENSIONALE ---
  template <typename Predicate>
  void integrate_gaussian_exact(int q, GaussianQuadContext<> &ctx,
                                Predicate &&is_inside) {
    if (k == N) {
      // Punto medio N-dimensionale per il check del predicato
      algoim::uvector<real, N> u_mid;
      for (int dim = 0; dim < N; ++dim)
        u_mid(dim) = 0.5;

      if (!is_inside(u_mid, 1.0))
        return;

      double val = 1.0;
      for (int dim = 0; dim < N; ++dim) {
        val *= gaussian_integral_1d(ctx.z_min[dim], ctx.z_max[dim]);
      }
      ctx.total_poc += val;
      return;
    }

    int max_count = 2;
    for (size_t i = 0; i < static_cast<size_t>(phi.count()); ++i)
      max_count += phi.poly(i).ext(k) - 1;

    auto integrand = [&](const uvector<real, N - 1> &xbase, real w) {
      real *nodes;
      algoim_spark_alloc(real, &nodes, max_count);
      nodes[0] = 0.0;
      nodes[1] = 1.0;
      int count = 2;

      for (size_t i = 0; i < static_cast<size_t>(phi.count()); ++i) {
        const auto &p = phi.poly(i);
        const auto &mask = phi.mask(i);
        int P = p.ext(k);

        if (!detail::lineIntersectsMask(mask, xbase, k))
          continue;

        real *pline, *roots;
        algoim_spark_alloc(real, &pline, P, &roots, P - 1);
        bernstein::collapseAlongAxis(p, xbase, k, pline);
        int rcount = bernstein::bernsteinUnitIntervalRealRoots(pline, P, roots);

        for (int j = 0; j < rcount; ++j) {
          auto x = add_component(xbase, k, roots[j]);
          if (detail::pointWithinMask(mask, x))
            nodes[count++] = roots[j];
        }
      };

      std::sort(nodes, nodes + count);
      assert(nodes[0] == real(0) && nodes[count - 1] == real(1));

      real tol = 10.0 * std::numeric_limits<real>::epsilon();
      using std::abs;
      for (int i = 1; i < count - 1; ++i)
        if (abs(nodes[i]) < tol)
          nodes[i] = 0.0;
        else if (abs(nodes[i] - 1) < tol)
          nodes[i] = 1.0;
        else if (abs(nodes[i] - nodes[i + 1]) < tol)
          nodes[i + 1] = nodes[i];

      for (int i = 0; i < count - 1; ++i) {
        real x0 = nodes[i];
        real x1 = nodes[i + 1];
        if (x0 == x1)
          continue;

        // Punto medio 1D convertito in punto N-D per il test
        real x_mid = x0 + (x1 - x0) * 0.5;
        algoim::uvector<real, N> u_mid = add_component(xbase, k, x_mid);
        if (!is_inside(u_mid, w))
          continue;

        // Coordinate N-D proiettate
        uvector<real, N> u_full = add_component(xbase, k, 0.0);
        double pdf_external = 1.0;
        double jacobian_external = 1.0;
        constexpr double INV_SQRT_2PI = 0.3989422804014327;

        for (int dim = 0; dim < N; ++dim) {
          if (dim == k)
            continue;
          double z_val =
              ctx.z_min[dim] + u_full(dim) * (ctx.z_max[dim] - ctx.z_min[dim]);
          pdf_external *= INV_SQRT_2PI * std::exp(-0.5 * z_val * z_val);
          jacobian_external *= (ctx.z_max[dim] - ctx.z_min[dim]);
        }

        double z0 = ctx.z_min[k] + x0 * (ctx.z_max[k] - ctx.z_min[k]);
        double z1 = ctx.z_min[k] + x1 * (ctx.z_max[k] - ctx.z_min[k]);

        double exact_k = gaussian_integral_1d(z0, z1);
        ctx.total_poc += w * pdf_external * jacobian_external * exact_k;
      }
    };

    base.integrate(AlwaysGL, q, integrand);
  }
};

template <> struct ImplicitPolyQuadrature<0> {};
} // namespace algoim

#endif