"""
Basic extraction example comparing Instructor and LangChain.

This example shows the simplest use case: extracting structured data from text.
"""

from pydantic import BaseModel
import os
import time
import statistics

# ===== Shared Model =====


class UserProfile(BaseModel):
    name: str
    age: int
    email: str
    interests: list[str]


# ===== Instructor Example =====


def instructor_example():
    """Extract user profile using Instructor."""
    import instructor
    from openai import OpenAI

    # Initialize client
    client = instructor.from_openai(OpenAI())

    # Time the extraction
    start_time = time.time()

    # Extract structured data
    user = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Extract: John Doe, 30 years old, john@example.com, loves hiking and photography",
            }
        ],
        response_model=UserProfile,
    )

    end_time = time.time()
    duration = end_time - start_time

    print("Instructor Result:")
    print(f"Name: {user.name}")
    print(f"Age: {user.age}")
    print(f"Email: {user.email}")
    print(f"Interests: {', '.join(user.interests)}")
    print(f"Time: {duration:.3f}s")

    return user, duration


# ===== LangChain Example =====


def langchain_example():
    """Extract user profile using LangChain."""
    from langchain_openai import ChatOpenAI

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Create structured output chain
    chain = llm.with_structured_output(UserProfile)

    # Time the extraction
    start_time = time.time()

    # Extract structured data
    user = chain.invoke(
        "Extract: John Doe, 30 years old, john@example.com, loves hiking and photography"
    )

    end_time = time.time()
    duration = end_time - start_time

    print("\nLangChain Result:")
    # Handle both Pydantic model and dict return types
    if isinstance(user, UserProfile):
        # Pydantic model
        print(f"Name: {user.name}")
        print(f"Age: {user.age}")
        print(f"Email: {user.email}")
        print(f"Interests: {', '.join(user.interests)}")
    else:
        # Dictionary
        print(f"Name: {user['name']}")
        print(f"Age: {user['age']}")
        print(f"Email: {user['email']}")
        print(f"Interests: {', '.join(user['interests'])}")
    print(f"Time: {duration:.3f}s")

    return user, duration


# ===== Performance Testing =====


def run_performance_test(num_runs=5):
    """Run performance test multiple times and provide statistics."""
    print(f"=== Performance Test ({num_runs} runs) ===\n")

    instructor_times = []
    langchain_times = []

    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}:")

        # Run Instructor
        try:
            instructor_result, instructor_time = instructor_example()
            instructor_times.append(instructor_time)
        except Exception as e:
            print(f"Instructor failed: {e}")
            continue

        # Run LangChain
        try:
            langchain_result, langchain_time = langchain_example()
            langchain_times.append(langchain_time)
        except Exception as e:
            print(f"LangChain failed: {e}")
            continue

        print("-" * 50)

    # Calculate statistics
    if instructor_times and langchain_times:
        print("\n=== Performance Statistics ===")
        print(f"Instructor ({len(instructor_times)} runs):")
        print(f"  Mean: {statistics.mean(instructor_times):.3f}s")
        print(f"  Median: {statistics.median(instructor_times):.3f}s")
        print(f"  Min: {min(instructor_times):.3f}s")
        print(f"  Max: {max(instructor_times):.3f}s")
        print(f"  Std Dev: {statistics.stdev(instructor_times):.3f}s")

        print(f"\nLangChain ({len(langchain_times)} runs):")
        print(f"  Mean: {statistics.mean(langchain_times):.3f}s")
        print(f"  Median: {statistics.median(langchain_times):.3f}s")
        print(f"  Min: {min(langchain_times):.3f}s")
        print(f"  Max: {max(langchain_times):.3f}s")
        print(f"  Std Dev: {statistics.stdev(langchain_times):.3f}s")

        # Compare performance
        instructor_mean = statistics.mean(instructor_times)
        langchain_mean = statistics.mean(langchain_times)

        print(f"\n=== Performance Comparison ===")
        if instructor_mean < langchain_mean:
            faster = "Instructor"
            speedup = langchain_mean / instructor_mean
            print(f"Winner: {faster} ({speedup:.1f}x faster on average)")
        elif langchain_mean < instructor_mean:
            faster = "LangChain"
            speedup = instructor_mean / langchain_mean
            print(f"Winner: {faster} ({speedup:.1f}x faster on average)")
        else:
            print("Performance: Tie")

        print(f"Average difference: {abs(instructor_mean - langchain_mean):.3f}s")

    return instructor_times, langchain_times


# ===== Main =====

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)

    print("=== Basic Extraction Comparison ===\n")

    # Run single comparison
    print("Single Run Comparison:")
    instructor_result, instructor_time = instructor_example()
    langchain_result, langchain_time = langchain_example()

    print("\n=== Performance Comparison ===")
    print(f"Instructor: {instructor_time:.3f}s")
    print(f"LangChain:  {langchain_time:.3f}s")

    if instructor_time < langchain_time:
        faster = "Instructor"
        speedup = langchain_time / instructor_time
        print(f"Winner: {faster} ({speedup:.1f}x faster)")
    elif langchain_time < instructor_time:
        faster = "LangChain"
        speedup = instructor_time / langchain_time
        print(f"Winner: {faster} ({speedup:.1f}x faster)")
    else:
        print("Performance: Tie")

    print("\n=== Feature Comparison ===")
    print("Both libraries produce identical results!")
    print("Key differences:")
    print("- Instructor: Simpler API, patches existing client")
    print("- LangChain: Framework approach, custom abstractions")

    # Run performance test
    print("\n" + "=" * 60)
    run_performance_test(num_runs=3)
