#!/usr/bin/env python3
"""
Benchmark script to verify claims made in the Instructor vs LangChain comparison blog post.

This script measures:
1. Performance comparison (latency)
2. Memory usage comparison
3. Token usage comparison
4. Code complexity comparison

Usage:
    python run.py --benchmark all
    python run.py --benchmark performance
    python run.py --benchmark memory
    python run.py --benchmark tokens
    python run.py --benchmark complexity

Requirements:
    pip install instructor langchain langchain-openai psutil tiktoken
"""

import time
import os
import sys
import argparse
from typing import Optional, Any  # noqa: UP035
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

# Optional imports with graceful fallbacks
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Test data
SAMPLE_TEXTS = [
    "John Smith is 25 years old and works as a software engineer",
    "Sarah Johnson, 32, is a marketing manager at TechCorp",
    "Mike Davis is 28 and currently studying computer science",
    "Lisa Chen, age 35, works as a data scientist",
    "Tom Wilson is 22 years old and recently graduated",
    "Emma Brown, 29, is a product manager at StartupXYZ",
    "Alex Garcia is 31 and works in sales",
    "Rachel Lee, 27, is a UX designer",
    "David Park is 33 and works as a project manager",
    "Jennifer Wu, age 26, is a software developer",
]


# Models for testing
class User(BaseModel):
    """Simple user extraction model"""

    name: str = Field(..., description="Full name of the person")
    age: int = Field(..., description="Age of the person")
    profession: Optional[str] = Field(None, description="Job title or profession")


class SimpleExtraction(BaseModel):
    """Model from blog post example"""

    items: list[str]
    count: int


class TaskExtraction(BaseModel):
    """Complex extraction from blog post"""

    tasks: list[str] = Field(..., description="list of actionable tasks")
    priorities: list[str] = Field(..., description="Priority for each task")
    deadline: Optional[str] = Field(None, description="Any mentioned deadlines")

    def get_high_priority_tasks(self) -> list[tuple]:
        return [
            (task, priority)
            for task, priority in zip(self.tasks, self.priorities)
            if priority.lower() == "high"
        ]

    @field_validator("priorities")
    def validate_priorities_length(cls, v, values):
        if "tasks" in values and len(v) != len(values["tasks"]):
            raise ValueError("Number of priorities must match number of tasks")
        return v


def get_memory_usage():
    """Get current memory usage in MB"""
    if not PSUTIL_AVAILABLE:
        print("⚠️  psutil not available, cannot measure memory usage")
        return 0.0

    process = psutil.Process(os.getpid())  # type: ignore
    return process.memory_info().rss / 1024 / 1024


def count_tokens(text: str) -> int:
    """Estimate token count using tiktoken"""
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))
    except ImportError:
        # Rough estimation if tiktoken not available
        return int(len(text.split()) * 1.3)


class InstructorBenchmark:
    """Benchmark Instructor performance"""

    def __init__(self):
        try:
            import instructor
            from openai import OpenAI

            self.client = instructor.from_openai(
                OpenAI(api_key=os.getenv("OPENAI_API_KEY", "fake-key-for-testing"))
            )
            self.available = True
        except ImportError as e:
            print(f"Instructor not available: {e}")
            self.available = False

    def simple_extraction(self, texts: list[str]) -> tuple[float, list[User]]:
        """Benchmark simple user extraction"""
        if not self.available:
            return 0.0, []

        start = time.time()
        results = []

        for text in texts:
            try:
                result = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    response_model=User,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Extract user information from: {text}",
                        }
                    ],
                )
                results.append(result)
            except Exception as e:
                print(f"Instructor extraction failed: {e}")
                # Create dummy result for comparison
                results.append(User(name="Unknown", age=0))

        elapsed = time.time() - start
        return elapsed, results

    def get_token_usage(self, text: str) -> int:
        """Get token count for Instructor approach"""
        system_prompt = "You are a helpful assistant."
        user_prompt = f"Extract user information from: {text}"

        total_tokens = count_tokens(system_prompt) + count_tokens(user_prompt)
        return total_tokens


class LangChainBenchmark:
    """Benchmark LangChain performance"""

    def __init__(self):
        try:
            from langchain.output_parsers import PydanticOutputParser
            from langchain.prompts import PromptTemplate
            from langchain_openai import ChatOpenAI

            self.parser = PydanticOutputParser(pydantic_object=User)
            self.prompt = PromptTemplate(
                template="Extract user information from: {text}\n{format_instructions}",
                input_variables=["text"],
                partial_variables={
                    "format_instructions": self.parser.get_format_instructions()
                },
            )
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=os.getenv("OPENAI_API_KEY", "fake-key-for-testing"),
            )
            self.chain = self.prompt | self.llm | self.parser
            self.available = True
        except ImportError as e:
            print(f"LangChain not available: {e}")
            self.available = False

    def simple_extraction(self, texts: list[str]) -> tuple[float, list[User]]:
        """Benchmark simple user extraction"""
        if not self.available:
            return 0.0, []

        start = time.time()
        results = []

        for text in texts:
            try:
                result = self.chain.invoke({"text": text})
                results.append(result)
            except Exception as e:
                print(f"LangChain extraction failed: {e}")
                # Create dummy result for comparison
                results.append(User(name="Unknown", age=0))

        elapsed = time.time() - start
        return elapsed, results

    def get_token_usage(self, text: str) -> int:
        """Get token count for LangChain approach"""
        format_instructions = self.parser.get_format_instructions()
        full_prompt = self.prompt.format(
            text=text, format_instructions=format_instructions
        )

        return count_tokens(full_prompt)


def benchmark_performance(num_iterations: int = 10) -> dict[str, Any]:
    """Benchmark performance comparison"""
    print(f"\n🏁 Performance Benchmark ({num_iterations} iterations)")
    print("=" * 50)

    instructor_bench = InstructorBenchmark()
    langchain_bench = LangChainBenchmark()

    results = {
        "instructor": {
            "available": instructor_bench.available,
            "time": 0.0,
            "results": [],
        },
        "langchain": {
            "available": langchain_bench.available,
            "time": 0.0,
            "results": [],
        },
    }

    # Test data subset for performance test
    test_texts = SAMPLE_TEXTS[:num_iterations]

    # Benchmark Instructor
    if instructor_bench.available:
        print("Testing Instructor...")
        time_taken, extracted_results = instructor_bench.simple_extraction(test_texts)
        results["instructor"]["time"] = time_taken
        results["instructor"]["results"] = extracted_results
        print(f"  Time: {time_taken:.2f}s")
        print(f"  Rate: {len(test_texts) / time_taken:.2f} extractions/sec")

    # Benchmark LangChain
    if langchain_bench.available:
        print("Testing LangChain...")
        time_taken, extracted_results = langchain_bench.simple_extraction(test_texts)
        results["langchain"]["time"] = time_taken
        results["langchain"]["results"] = extracted_results
        print(f"  Time: {time_taken:.2f}s")
        print(f"  Rate: {len(test_texts) / time_taken:.2f} extractions/sec")

    # Compare results
    if results["instructor"]["available"] and results["langchain"]["available"]:
        instructor_time = results["instructor"]["time"]
        langchain_time = results["langchain"]["time"]

        if instructor_time > 0 and langchain_time > 0:
            speedup = (langchain_time - instructor_time) / instructor_time * 100
            print(f"\n📊 Performance Comparison:")
            print(f"  Instructor: {instructor_time:.2f}s")
            print(f"  LangChain:  {langchain_time:.2f}s")
            print(
                f"  Difference: {speedup:+.1f}% ({'faster' if speedup < 0 else 'slower'} than Instructor)"
            )

            # Blog post claims LangChain is 17% slower
            blog_claim = 17.0
            print(f"  Blog claim: LangChain is {blog_claim}% slower")
            print(
                f"  Actual:     LangChain is {-speedup:.1f}% {'faster' if speedup < 0 else 'slower'}"
            )

    return results


def benchmark_memory() -> dict[str, float]:
    """Benchmark memory usage comparison"""
    print(f"\n💾 Memory Usage Benchmark")
    print("=" * 50)

    if not PSUTIL_AVAILABLE:
        print("⚠️  psutil not available - cannot run memory benchmark")
        print("   Install with: pip install psutil")
        return {"baseline": 0.0, "instructor": 0.0, "langchain": 0.0}

    # Baseline memory
    baseline_memory = get_memory_usage()
    print(f"Baseline memory: {baseline_memory:.1f} MB")

    results = {"baseline": baseline_memory}

    # Test Instructor memory usage
    print("\nTesting Instructor memory usage...")
    instructor_start = get_memory_usage()

    try:
        import instructor
        from openai import OpenAI

        client = instructor.from_openai(OpenAI(api_key="fake-key"))
        instructor_end = get_memory_usage()
        instructor_usage = instructor_end - baseline_memory
        results["instructor"] = instructor_usage
        print(f"  Instructor imports: +{instructor_usage:.1f} MB")
    except ImportError as e:
        print(f"  Instructor not available: {e}")
        results["instructor"] = 0.0

    # Test LangChain memory usage
    print("Testing LangChain memory usage...")
    langchain_start = get_memory_usage()

    try:
        from langchain.output_parsers import PydanticOutputParser
        from langchain.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI
        from langchain.schema import BaseOutputParser

        langchain_end = get_memory_usage()
        langchain_usage = langchain_end - baseline_memory
        results["langchain"] = langchain_usage
        print(f"  LangChain imports: +{langchain_usage:.1f} MB")
    except ImportError as e:
        print(f"  LangChain not available: {e}")
        results["langchain"] = 0.0

    # Compare memory usage
    if results["instructor"] > 0 and results["langchain"] > 0:
        ratio = results["langchain"] / results["instructor"]
        print(f"\n📊 Memory Comparison:")
        print(f"  Instructor: {results['instructor']:.1f} MB")
        print(f"  LangChain:  {results['langchain']:.1f} MB")
        print(f"  Ratio:      {ratio:.1f}x (LangChain uses {ratio:.1f}x more memory)")

        # Blog post claims Instructor ~15MB, LangChain ~85MB
        print(f"  Blog claim: Instructor ~15MB, LangChain ~85MB")
        print(f"  Blog ratio: {85 / 15:.1f}x")

    return results


def benchmark_tokens() -> dict[str, Any]:
    """Benchmark token usage comparison"""
    print(f"\n🎯 Token Usage Benchmark")
    print("=" * 50)

    instructor_bench = InstructorBenchmark()
    langchain_bench = LangChainBenchmark()

    results = {"instructor": [], "langchain": []}

    sample_text = "John Smith is 25 years old and works as a software engineer"

    if instructor_bench.available:
        instructor_tokens = instructor_bench.get_token_usage(sample_text)
        results["instructor"] = instructor_tokens
        print(f"Instructor tokens: {instructor_tokens}")

    if langchain_bench.available:
        langchain_tokens = langchain_bench.get_token_usage(sample_text)
        results["langchain"] = langchain_tokens
        print(f"LangChain tokens:  {langchain_tokens}")

        if results["instructor"] > 0:
            savings = (langchain_tokens - instructor_tokens) / langchain_tokens * 100
            print(f"\n📊 Token Comparison:")
            print(f"  Instructor: {instructor_tokens} tokens")
            print(f"  LangChain:  {langchain_tokens} tokens")
            print(f"  Difference: {langchain_tokens - instructor_tokens} tokens")
            print(
                f"  Savings:    {savings:.1f}% (Instructor uses {savings:.1f}% fewer tokens)"
            )

            # Blog post claims 10-20% savings
            print(f"  Blog claim: Instructor saves 10-20% tokens")

    return results


def benchmark_complexity():
    """Analyze code complexity comparison"""
    print(f"\n🔧 Code Complexity Analysis")
    print("=" * 50)

    instructor_code = """
# Instructor approach
import instructor
from pydantic import BaseModel
from openai import OpenAI

class User(BaseModel):
    name: str
    age: int

client = instructor.from_openai(OpenAI())
user = client.chat.completions.create(
    model="gpt-4",
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 25"}]
)
"""

    langchain_code = """
# LangChain approach
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

parser = PydanticOutputParser(pydantic_object=User)
prompt = PromptTemplate(
    template="Extract user info: {text}\\n{format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | parser
user = chain.invoke({"text": "John is 25"})
"""

    instructor_lines = len(
        [
            l
            for l in instructor_code.strip().split("\n")
            if l.strip() and not l.strip().startswith("#")
        ]
    )
    langchain_lines = len(
        [
            l
            for l in langchain_code.strip().split("\n")
            if l.strip() and not l.strip().startswith("#")
        ]
    )

    print(f"Lines of code (excluding comments):")
    print(f"  Instructor: {instructor_lines} lines")
    print(f"  LangChain:  {langchain_lines} lines")
    print(
        f"  Difference: {langchain_lines - instructor_lines} more lines for LangChain"
    )

    instructor_imports = instructor_code.count("import") + instructor_code.count("from")
    langchain_imports = langchain_code.count("import") + langchain_code.count("from")

    print(f"\nImport statements:")
    print(f"  Instructor: {instructor_imports} imports")
    print(f"  LangChain:  {langchain_imports} imports")

    print(f"\nComplexity Analysis:")
    print(f"  Instructor: Direct API call, minimal setup")
    print(f"  LangChain:  Chain composition, parser setup, prompt templates")

    return {
        "instructor_lines": instructor_lines,
        "langchain_lines": langchain_lines,
        "instructor_imports": instructor_imports,
        "langchain_imports": langchain_imports,
    }


def run_all_benchmarks():
    """Run all benchmarks"""
    print("🚀 Instructor vs LangChain Benchmark Suite")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "fake-key-for-testing":
        print("\n⚠️  Warning: No OpenAI API key found. Some tests will use mock data.")
        print("   Set OPENAI_API_KEY environment variable for live testing.")

    all_results = {}

    # Run benchmarks
    all_results["performance"] = benchmark_performance(10)
    all_results["memory"] = benchmark_memory()
    all_results["tokens"] = benchmark_tokens()
    all_results["complexity"] = benchmark_complexity()

    print(f"\n📋 Summary Report")
    print("=" * 50)

    # Performance summary
    perf = all_results["performance"]
    if perf["instructor"]["available"] and perf["langchain"]["available"]:
        inst_time = perf["instructor"]["time"]
        lang_time = perf["langchain"]["time"]
        if inst_time > 0 and lang_time > 0:
            speedup = (lang_time - inst_time) / inst_time * 100
            print(
                f"⚡ Performance: LangChain is {-speedup:.1f}% {'faster' if speedup < 0 else 'slower'}"
            )

    # Memory summary
    mem = all_results["memory"]
    if mem.get("instructor", 0) > 0 and mem.get("langchain", 0) > 0:
        ratio = mem["langchain"] / mem["instructor"]
        print(f"💾 Memory: LangChain uses {ratio:.1f}x more memory")

    # Token summary
    tokens = all_results["tokens"]
    if isinstance(tokens["instructor"], int) and isinstance(tokens["langchain"], int):
        if tokens["langchain"] > 0:
            savings = (
                (tokens["langchain"] - tokens["instructor"]) / tokens["langchain"] * 100
            )
            print(f"🎯 Tokens: Instructor saves {savings:.1f}% tokens")

    # Complexity summary
    comp = all_results["complexity"]
    line_diff = comp["langchain_lines"] - comp["instructor_lines"]
    print(f"🔧 Complexity: LangChain requires {line_diff} more lines of code")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Instructor vs LangChain")
    parser.add_argument(
        "--benchmark",
        choices=["all", "performance", "memory", "tokens", "complexity"],
        default="all",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations for performance test",
    )

    args = parser.parse_args()

    if args.benchmark == "all":
        run_all_benchmarks()
    elif args.benchmark == "performance":
        benchmark_performance(args.iterations)
    elif args.benchmark == "memory":
        benchmark_memory()
    elif args.benchmark == "tokens":
        benchmark_tokens()
    elif args.benchmark == "complexity":
        benchmark_complexity()


if __name__ == "__main__":
    main()
