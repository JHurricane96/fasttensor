// The original file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Original work Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Original work Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
// Modified work Copyright (C) 2019 Arun Ramachandran <ramachandran.arun@outlook.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <array>

#if defined(_WIN32) || defined(__CYGWIN__)
# ifndef NOMINMAX
#   define NOMINMAX
#   define FASTTENSOR_BT_UNDEF_NOMINMAX
# endif
# ifndef WIN32_LEAN_AND_MEAN
#   define WIN32_LEAN_AND_MEAN
#   define FASTTENSOR_BT_UNDEF_WIN32_LEAN_AND_MEAN
# endif
# include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach_time.h>
#else
# include <unistd.h>
# include <time.h>
#endif

static void escape(void *p) {
  asm volatile("" : : "g"(p) : "memory");
}

static void clobber() {
  asm volatile("" : : : "memory");
}

enum {
  CPU_TIMER = 0,
  REAL_TIMER = 1
};

/** Elapsed time timer keeping the best try.
  *
  * On POSIX platforms we use clock_gettime with CLOCK_PROCESS_CPUTIME_ID.
  * On Windows we use QueryPerformanceCounter
  *
  * Important: on linux, you must link with -lrt
  */
class BenchTimer
{
public:

  BenchTimer()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    m_frequency = (double)freq.QuadPart;
#endif
    reset();
  }

  ~BenchTimer() {}

  inline void reset()
  {
    m_bests.fill(1e9);
    m_worsts.fill(0);
    m_totals.fill(0);
    m_squared_totals.fill(0);
  }
  inline void start()
  {
    m_starts[CPU_TIMER]  = getCpuTime();
    m_starts[REAL_TIMER] = getRealTime();
  }
  inline void stop()
  {
    m_times[CPU_TIMER] = getCpuTime() - m_starts[CPU_TIMER];
    m_times[REAL_TIMER] = getRealTime() - m_starts[REAL_TIMER];
    m_bests[0] = std::min(m_bests[0],m_times[0]);
    m_bests[1] = std::min(m_bests[1],m_times[1]);
    m_worsts[0] = std::max(m_worsts[0],m_times[0]);
    m_worsts[1] = std::max(m_worsts[1],m_times[1]);
    m_totals[0] += m_times[0];
    m_totals[1] += m_times[1];
    m_squared_totals[0] += m_times[0] * m_times[0];
    m_squared_totals[1] += m_times[1] * m_times[1];
  }

  /** Return the elapsed time in seconds between the last start/stop pair
    */
  inline double value(int TIMER = CPU_TIMER) const
  {
    return m_times[TIMER];
  }

  /** Return the best elapsed time in seconds
    */
  inline double best(int TIMER = CPU_TIMER) const
  {
    return m_bests[TIMER];
  }

  /** Return the worst elapsed time in seconds
    */
  inline double worst(int TIMER = CPU_TIMER) const
  {
    return m_worsts[TIMER];
  }

  /** Return the total elapsed time in seconds.
    */
  inline double total(int TIMER = CPU_TIMER) const
  {
    return m_totals[TIMER];
  }

  /** Return the total of squares of elapsed time in seconds.
    */
  inline double squared_total(int TIMER = CPU_TIMER) const
  {
    return m_squared_totals[TIMER];
  }

  inline double getCpuTime() const
  {
#ifdef _WIN32
    LARGE_INTEGER query_ticks;
    QueryPerformanceCounter(&query_ticks);
    return query_ticks.QuadPart/m_frequency;
#elif __APPLE__
    return double(mach_absolute_time())*1e-9;
#else
    timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
#endif
  }

  inline double getRealTime() const
  {
#ifdef _WIN32
    SYSTEMTIME st;
    GetSystemTime(&st);
    return (double)st.wSecond + 1.e-3 * (double)st.wMilliseconds;
#elif __APPLE__
    return double(mach_absolute_time())*1e-9;
#else
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return double(ts.tv_sec) + 1e-9 * double(ts.tv_nsec);
#endif
  }

protected:
#if defined(_WIN32) || defined(__CYGWIN__)
  double m_frequency;
#endif
  std::array<double, 2> m_starts;
  std::array<double, 2> m_times;
  std::array<double, 2> m_bests;
  std::array<double, 2> m_worsts;
  std::array<double, 2> m_totals;
  std::array<double, 2> m_squared_totals;
};

#define BENCH(TIMER,TRIES,REP,CODE) { \
    TIMER.reset(); \
    for(int uglyvarname1=0; uglyvarname1<TRIES; ++uglyvarname1){ \
      TIMER.start(); \
      for(int uglyvarname2=0; uglyvarname2<REP; ++uglyvarname2){ \
        CODE; \
      } \
      TIMER.stop(); \
      clobber(); \
    } \
  }

// clean #defined tokens
#ifdef FASTTENSOR_BT_UNDEF_NOMINMAX
# undef FASTTENSOR_BT_UNDEF_NOMINMAX
# undef NOMINMAX
#endif

#ifdef FASTTENSOR_BT_UNDEF_WIN32_LEAN_AND_MEAN
# undef FASTTENSOR_BT_UNDEF_WIN32_LEAN_AND_MEAN
# undef WIN32_LEAN_AND_MEAN
#endif
