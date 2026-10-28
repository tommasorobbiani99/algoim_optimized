#ifndef ALGOIM_SPARKSTACK_HPP
#define ALGOIM_SPARKSTACK_HPP

#include "uvector.hpp"
#include <cstdlib>
#include <iostream>
#include <new>
#include <type_traits>
#include <typeinfo>

namespace algoim {
template <typename T, int N> class xarray;

template <typename T> class SparkStack {
  static size_t get_capacity() {
    static const size_t cap = []() -> size_t {
      if constexpr (std::is_same_v<T, double>) {
        if (const char *env_mb = std::getenv("ALGOIM_SPARK_MB")) {
          size_t mb = std::strtoull(env_mb, nullptr, 10);
          if (mb > 0)
            return (mb * 1024ULL * 1024ULL) / sizeof(double);
        }
        return 1ULL << 29; // 4 GB virtuali per thread (allocati solo on-demand)
      } else {
        return 1ULL << 24; // ~16M elementi per int/bool
      }
    }();
    return cap;
  }

  struct LazyThreadBuffer {
    T *ptr = nullptr;
    explicit LazyThreadBuffer(size_t n) {
      ptr = static_cast<T *>(std::malloc(n * sizeof(T)));
      if (!ptr) {
        std::cerr << "[SparkStack] Impossibile riservare "
                  << (n * sizeof(T)) / (1024 * 1024) << " MB virtuali.\n";
        throw std::bad_alloc();
      }
    }
    ~LazyThreadBuffer() { std::free(ptr); }
  };

  template <typename... R> static size_t alloc(T **ptr, size_t len, R... rest) {
    const size_t capacity = get_capacity();
    if (static_cast<size_t>(pos()) + len > capacity) {
      std::cerr << "SparkStack<T = " << typeid(T).name()
                << ">: capacity=" << capacity << " and pos=" << pos()
                << " insufficient for request len=" << len << '\n'
                << "    Usa 'export ALGOIM_SPARK_MB=8192' da terminale!\n";
      throw std::bad_alloc();
    }
    *ptr = base() + pos();
    pos() += len;
    if constexpr (sizeof...(rest) == 0)
      return len;
    else
      return len + alloc(rest...);
  }

  static T *base() {
    static thread_local LazyThreadBuffer buff(get_capacity());
    return buff.ptr;
  }

  static ptrdiff_t &pos() {
    static thread_local ptrdiff_t pos_ = 0;
    return pos_;
  }

  size_t len_;

  SparkStack(const SparkStack &) = delete;
  SparkStack(SparkStack &&) = delete;
  SparkStack &operator=(const SparkStack &) = delete;
  SparkStack &operator=(SparkStack &&) = delete;

public:
  template <typename... R>
  explicit SparkStack(T **ptr, size_t len, R &&...rest) {
    len_ = alloc(ptr, len, rest...);
  }

  template <typename... R>
  explicit SparkStack(T value, T **ptr, size_t len, R &&...rest) {
    T *start = base() + pos();
    len_ = alloc(ptr, len, rest...);
    for (size_t i = 0; i < len_; ++i)
      *(start + i) = value;
  }

  template <int N>
  explicit SparkStack(uvector<T *, N> &ptr, const uvector<int, N> &ext) {
    len_ = 0;
    for (int i = 0; i < N; ++i)
      len_ += alloc(&ptr(i), ext(i));
  }

  template <int... N> explicit SparkStack(xarray<T, N> &...a) {
    len_ = (alloc(&a.data_, a.size()) + ...);
  }

  ~SparkStack() { pos() -= len_; }
};

#define algoim_CONCAT2(x, y) x##y
#define algoim_CONCAT(x, y) algoim_CONCAT2(x, y)
#define algoim_spark_alloc(T, ...)                                             \
  SparkStack<T> algoim_CONCAT(spark_alloc_var_, __LINE__)(__VA_ARGS__)
#define algoim_spark_alloc_def(T, val, ...)                                    \
  SparkStack<T> algoim_CONCAT(spark_alloc_var_, __LINE__)(val, __VA_ARGS__)
#define algoim_spark_alloc_vec(T, ptr, ext)                                    \
  SparkStack<T> algoim_CONCAT(spark_alloc_var_, __LINE__)(ptr, ext)

} // namespace algoim

#endif
