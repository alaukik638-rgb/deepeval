# # 1. Length-based scoring
# def generate_hypothetical_score(test_case) -> float | None:
#     output_length = len(test_case.actual_output)
#     if output_length < 10:
#         return 0.3
#     elif output_length < 50:
#         return 0.7
#     else:
#         return 0.9
#
# # 2. Keyword matching
# def generate_hypothetical_score(test_case) -> float | None:
#     keywords = ['result', 'answer', 'solution']
#     count = sum(1 for kw in keywords if kw in test_case.actual_output.lower())
#     return min(count / len(keywords), 1.0)
#
# # 3. Simple exact match
# def generate_hypothetical_score(test_case) -> float | None:
#     return 1.0 if test_case.expected_output in test_case.actual_output else 0.0
#
#
# # 4. Similarity ratio (requires: pip install python-Levenshtein)
# from difflib import SequenceMatcher
#
#
# def generate_hypothetical_score(test_case) -> float | None:
#     similarity = SequenceMatcher(None,
#                                  test_case.expected_output.lower(),
#                                  test_case.actual_output.lower()).ratio()
#     return similarity
#
#
# def generate_hypothetical_reason(test_case) -> str | None:
#     score = generate_hypothetical_score(test_case)
#     if score > 0.8:
#         return "High similarity between expected and actual output"
#     elif score > 0.5:
#         return "Moderate similarity, some differences found"
#     else:
#         return "Low similarity, outputs differ significantly"
#
#
# # 5. Numeric answer extraction and comparison
# import re
#
#
# def generate_hypothetical_score(test_case) -> float | None:
#     expected_nums = re.findall(r'-?\d+\.?\d*', test_case.expected_output)
#     actual_nums = re.findall(r'-?\d+\.?\d*', test_case.actual_output)
#
#     if not expected_nums or not actual_nums:
#         return 0.5
#
#     expected_val = float(expected_nums[0])
#     actual_val = float(actual_nums[0])
#
#     # Check if within 5% tolerance
#     tolerance = abs(expected_val * 0.05)
#     if abs(expected_val - actual_val) <= tolerance:
#         return 1.0
#     else:
#         return max(0.0, 1.0 - abs(expected_val - actual_val) / expected_val)
#
#
# # 6. Sentiment-based scoring (requires: pip install textblob)
# from textblob import TextBlob
#
#
# def generate_hypothetical_score(test_case) -> float | None:
#     blob = TextBlob(test_case.actual_output)
#     polarity = blob.sentiment.polarity  # -1 to 1
#     # Normalize to 0-1
#     return (polarity + 1) / 2
#
#
# # 7. Embedding similarity (requires: pip install sentence-transformers)
# from sentence_transformers import SentenceTransformer, util
#
# model = SentenceTransformer('all-MiniLM-L6-v2')
#
#
# def generate_hypothetical_score(test_case) -> float | None:
#     emb1 = model.encode(test_case.expected_output, convert_to_tensor=True)
#     emb2 = model.encode(test_case.actual_output, convert_to_tensor=True)
#     similarity = util.cos_sim(emb1, emb2).item()
#     return max(0.0, min(1.0, similarity))
#
#
# # 8. Multi-criteria weighted scoring
# def generate_hypothetical_score(test_case) -> float | None:
#     # Criterion 1: Length appropriateness (30%)
#     expected_len = len(test_case.expected_output.split())
#     actual_len = len(test_case.actual_output.split())
#     length_score = 1.0 - min(abs(expected_len - actual_len) / max(expected_len, 1), 1.0)
#
#     # Criterion 2: Keyword coverage (30%)
#     expected_words = set(test_case.expected_output.lower().split())
#     actual_words = set(test_case.actual_output.lower().split())
#     keyword_score = len(expected_words & actual_words) / len(expected_words) if expected_words else 0
#
#     # Criterion 3: Textual similarity (40%)
#     similarity = SequenceMatcher(None, test_case.expected_output, test_case.actual_output).ratio()
#
#     final_score = (length_score * 0.3) + (keyword_score * 0.3) + (similarity * 0.4)
#     return final_score
#
#
#
# # ***************************** To Be Continued ***********************************
#
# def generate_hypothetical_reason(test_case) -> str | None:
#     score = generate_hypothetical_score(test_case)
#     reasons = []
#
#     if score > 0.8:
#         reasons.append("Excellent match")
#     elif score > 0.6:
#         reasons.append("Good match with minor differences")
#     else:
#         reasons.append("Significant differences detected")
#
#     # Add specific feedback
#     expected_words = set(test_case.expected_output.lower().split())
#     actual_words = set(test_case.actual_output.lower().split())
#     missing = expected_words - actual_words
#
#     if missing:
#         reasons.append(f"Missing keywords: {', '.join(list(missing)[:3])}")
#
#     return " | ".join(reasons)
#
#
# # 9. Context-aware scoring with retrieval
# def generate_hypothetical_score(test_case) -> float | None:
#     # Check if actual output contains the context information
#     context_score = 0.0
#     if hasattr(test_case, 'context') and test_case.context:
#         context_words = set(test_case.context[0].lower().split())
#         actual_words = set(test_case.actual_output.lower().split())
#         context_score = len(context_words & actual_words) / len(context_words)
#
#     # Check if expected output is present
#     answer_score = 1.0 if test_case.expected_output.lower() in test_case.actual_output.lower() else 0.0
#
#     # Weighted combination
#     return (context_score * 0.3) + (answer_score * 0.7)
#
#
# # 10. LLM-as-judge pattern (requires: OpenAI API or similar)
# from openai import OpenAI
#
# client = OpenAI()
#
#
# def generate_hypothetical_score(test_case) -> float | None:
#     prompt = f"""
#     Evaluate the following answer on a scale of 0.0 to 1.0:
#
#     Question: {test_case.input}
#     Expected Answer: {test_case.expected_output}
#     Actual Answer: {test_case.actual_output}
#
#     Return only a float score between 0.0 and 1.0.
#     """
#
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0
#     )
#
#     try:
#         score = float(response.choices[0].message.content.strip())
#         return max(0.0, min(1.0, score))
#     except:
#         return 0.5
#
#
# def generate_hypothetical_reason(test_case) -> str | None:
#     prompt = f"""
#     Explain why this answer is good or bad:
#
#     Question: {test_case.input}
#     Expected Answer: {test_case.expected_output}
#     Actual Answer: {test_case.actual_output}
#
#     Provide a brief explanation (2-3 sentences).
#     """
#
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0
#     )
#
#     return response.choices[0].message.content.strip()
#
#
# # 11. Composite metric with async LLM calls
# import asyncio
# from openai import AsyncOpenAI
#
# async_client = AsyncOpenAI()
#
#
# async def async_generate_hypothetical_score(test_case) -> float | None:
#     # Run multiple evaluation criteria in parallel
#     tasks = [
#         evaluate_factuality(test_case),
#         evaluate_relevance(test_case),
#         evaluate_coherence(test_case)
#     ]
#
#     scores = await asyncio.gather(*tasks)
#     # Weighted average
#     return sum(scores) / len(scores)
#
#
# async def evaluate_factuality(test_case) -> float:
#     # Check if facts in actual output match expected
#     prompt = f"Rate factual accuracy (0.0-1.0): Expected: {test_case.expected_output}, Got: {test_case.actual_output}"
#     response = await async_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return float(response.choices[0].message.content.strip())
#
#
# async def evaluate_relevance(test_case) -> float:
#     # Similar pattern for relevance
#     pass
#
#
# async def evaluate_coherence(test_case) -> float:
#     # Similar pattern for coherence
#     pass