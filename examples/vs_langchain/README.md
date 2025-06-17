# Instructor vs LangChain Benchmark

This benchmark script verifies the performance claims made in the [Instructor vs LangChain comparison blog post](../../docs/blog/posts/instructor-vs-langchain-comparison.md).

## What it tests

1. **Performance**: Measures execution time for structured data extraction
2. **Memory Usage**: Compares memory footprint of imports and setup
3. **Token Usage**: Analyzes prompt token consumption differences
4. **Code Complexity**: Compares lines of code and setup complexity

## Setup

Install required dependencies:

```bash
pip install instructor langchain langchain-openai psutil tiktoken
```

Set your OpenAI API key (optional - script works with mock data too):

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

Run all benchmarks:

```bash
python run.py --benchmark all
```

Run specific benchmarks:

```bash
python run.py --benchmark performance
python run.py --benchmark memory
python run.py --benchmark tokens
python run.py --benchmark complexity
```

Adjust performance test iterations:

```bash
python run.py --benchmark performance --iterations 50
```

## Expected Results

Based on the blog post claims, you should see:

- **Performance**: LangChain ~17% slower than Instructor
- **Memory**: LangChain uses ~5.7x more memory than Instructor
- **Tokens**: Instructor saves 10-20% tokens vs LangChain
- **Complexity**: LangChain requires more setup code

## Sample Output

```
🚀 Instructor vs LangChain Benchmark Suite
============================================================
Timestamp: 2025-06-17T23:45:00

🏁 Performance Benchmark (10 iterations)
==================================================
Testing Instructor...
  Time: 12.34s
  Rate: 0.81 extractions/sec
Testing LangChain...
  Time: 14.45s
  Rate: 0.69 extractions/sec

📊 Performance Comparison:
  Instructor: 12.34s
  LangChain:  14.45s
  Difference: +17.1% (slower than Instructor)
  Blog claim: LangChain is 17.0% slower
  Actual:     LangChain is 17.1% slower

💾 Memory Usage Benchmark
==================================================
  Instructor imports: +18.2 MB
  LangChain imports:  +97.6 MB
  Ratio: 5.4x (LangChain uses 5.4x more memory)
```

## Notes

- Without an API key, the script uses mock data for timing comparisons
- Memory usage may vary based on system and Python version
- Token counts are estimated using tiktoken
- Results may vary based on network latency and API response times