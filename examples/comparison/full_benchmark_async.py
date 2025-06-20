"""
Async Full Benchmark: Instructor (all OpenAI modes) vs LangChain

Runs the same extraction task using Instructor (async) in all supported OpenAI modes and LangChain (async),
then compares their performance and reliability side-by-side.
"""

import os
import time
import statistics
import asyncio
from pydantic import BaseModel
import instructor
from openai import AsyncOpenAI

# ===== Shared Model =====


class UserProfile(BaseModel):
    name: str
    age: int
    email: str
    interests: list[str] = Field(
        default_factory=list, description="List of user interests in lowercase"
    )


# ===== Test Data =====

TEST_TEXT = (
    "Extract: John Doe, 30 years old, john@example.com, loves hiking and photography"
)
NUM_RUNS = 50
MODEL = "gpt-4o-mini"

# ===== Async Instructor Mode Benchmark =====


async def async_instructor_mode_benchmark(mode, num_runs=NUM_RUNS):
    client = instructor.from_openai(AsyncOpenAI(), mode=mode)
    timings = []
    successes = 0
    results = []

    async def run_once():
        start = time.perf_counter()
        try:
            user = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": TEST_TEXT}],
                response_model=UserProfile,
            )
            elapsed = time.perf_counter() - start
            return elapsed, user, True
        except Exception as e:
            print(f"Error in {mode.value}: {e}")
            return None, None, False

    # Run tasks with limited concurrency to avoid rate limits
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

    async def run_with_semaphore():
        async with semaphore:
            return await run_once()

    tasks = [run_with_semaphore() for _ in range(num_runs)]
    run_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in run_results:
        if isinstance(result, Exception):
            print(f"Task failed with exception: {result}")
            timings.append(None)
            results.append(None)
        else:
            elapsed, user, success = result
            if success:
                timings.append(elapsed)
                results.append(user)
                successes += 1
            else:
                timings.append(None)
                results.append(None)

    valid_timings = [t for t in timings if t is not None]
    return {
        "name": f"Instructor ({mode.value})",
        "mean": statistics.mean(valid_timings) if valid_timings else None,
        "median": statistics.median(valid_timings) if valid_timings else None,
        "success_rate": successes / num_runs * 100,
        "stddev": statistics.stdev(valid_timings) if len(valid_timings) > 1 else 0,
        "results": results,
    }


# ===== Async LangChain Benchmark =====


async def async_langchain_benchmark(num_runs=NUM_RUNS):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=MODEL)
    structured_llm = llm.with_structured_output(UserProfile)
    timings = []
    successes = 0
    results = []

    async def run_once():
        start = time.perf_counter()
        try:
            user = await structured_llm.ainvoke(TEST_TEXT)
            elapsed = time.perf_counter() - start
            return elapsed, user, True
        except Exception as e:
            print(f"Error in LangChain: {e}")
            return None, None, False

    # Run tasks with limited concurrency to avoid rate limits
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

    async def run_with_semaphore():
        async with semaphore:
            return await run_once()

    tasks = [run_with_semaphore() for _ in range(num_runs)]
    run_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in run_results:
        if isinstance(result, Exception):
            print(f"Task failed with exception: {result}")
            timings.append(None)
            results.append(None)
        else:
            elapsed, user, success = result
            if success:
                timings.append(elapsed)
                results.append(user)
                successes += 1
            else:
                timings.append(None)
                results.append(None)

    valid_timings = [t for t in timings if t is not None]
    return {
        "name": "LangChain",
        "mean": statistics.mean(valid_timings) if valid_timings else None,
        "median": statistics.median(valid_timings) if valid_timings else None,
        "success_rate": successes / num_runs * 100,
        "stddev": statistics.stdev(valid_timings) if len(valid_timings) > 1 else 0,
        "results": results,
    }


# ===== Main Async Benchmark Runner =====


async def main():
    print(f"Running async benchmark for {NUM_RUNS} runs per mode...\n")
    modes = [
        instructor.Mode.TOOLS,
        instructor.Mode.JSON,
        instructor.Mode.MD_JSON,
        instructor.Mode.TOOLS_STRICT,
        instructor.Mode.JSON_O1,
    ]
    all_results = []
    # Run Instructor modes
    for mode in modes:
        print(f"Benchmarking Instructor mode: {mode.value}")
        result = await async_instructor_mode_benchmark(mode)
        all_results.append(result)
    # Run LangChain
    print("Benchmarking LangChain async...")
    lc_result = await async_langchain_benchmark()
    all_results.append(lc_result)
    # Print summary table
    print("\nSummary Table:")
    print(
        f"{'Name':<30} {'Mean (s)':<10} {'Median (s)':<12} {'Success Rate':<14} {'Std Dev':<10}"
    )
    print("-" * 80)
    for r in all_results:
        success_rate_str = f"{r['success_rate']:.1f}%"
        mean_str = (
            f"{r['mean']:.3f}" if isinstance(r["mean"], float) else str(r["mean"])
        )
        median_str = (
            f"{r['median']:.3f}" if isinstance(r["median"], float) else str(r["median"])
        )
        stddev_str = (
            f"{r['stddev']:.3f}" if isinstance(r["stddev"], float) else str(r["stddev"])
        )
        print(
            f"{r['name']:<30} {mean_str:<10} {median_str:<12} {success_rate_str:<14} {stddev_str:<10}"
        )


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback

        print(f"\n❌ Async benchmark failed with error: {e}")
        print(f"Full traceback:")
        traceback.print_exc()
        exit(1)
