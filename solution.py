import re
import math
import collections
import numpy as np
from harness_base import Harness


class MyHarness(Harness):
    USE_ARBITER = False

    def __init__(self, call_llm, count_tokens, count_messages_tokens, max_prompt_tokens: int):
        super().__init__(call_llm, count_tokens, count_messages_tokens, max_prompt_tokens)

        self.doc_freqs = collections.Counter()
        self.postings = collections.defaultdict(list)

        self.docs = []
        self.doc_lens = []
        self.raw_texts = []
        self.labels = []

        self.avgdl = 0.0
        self.N = 0
        self.all_labels = set()

        self.exact_lookup = collections.defaultdict(collections.Counter)
        self.label_to_indices = collections.defaultdict(list)
        self.global_label_counts = collections.Counter()

    def extract(self, text: str) -> list:

        text = (text or "").lower()
        tokens = [m.group() for m in re.finditer(r'[a-z0-9]+|[\u4e00-\u9fa5]|[?？!！]', text)]
        bigrams = [f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def _norm_key(self, text: str) -> str:
        text = (text or "").lower()
        tokens = [m.group() for m in re.finditer(r'[a-z0-9]+|[\u4e00-\u9fa5]', text)]
        return "".join(tokens)

    def update(self, text: str, label: str) -> None:
        super().update(text, label)

        self.all_labels.add(label)
        self.global_label_counts[label] += 1

        idx = self.N
        raw_text = text or ""

        features = self.extract(raw_text)
        tf = collections.Counter(features)

        self.docs.append(features)
        self.doc_lens.append(len(features))
        self.raw_texts.append(raw_text)
        self.labels.append(label)
        self.label_to_indices[label].append(idx)
        self.exact_lookup[self._norm_key(raw_text)][label] += 1

        self.N += 1
        self.avgdl = ((self.avgdl * (self.N - 1)) + len(features)) / self.N

        for f in set(features):
            self.doc_freqs[f] += 1

        for f, c in tf.items():
            self.postings[f].append((idx, c))

    def _get_bm25_scores(self, query_features: list) -> np.ndarray:
        k1 = 1.2
        b = 0.5

        scores = np.zeros(self.N)
        if self.N == 0 or self.avgdl <= 0:
            return scores

        q_counts = collections.Counter(query_features)

        for f, q_tf in q_counts.items():
            if f not in self.doc_freqs:
                continue

            df = self.doc_freqs[f]
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            if idf < 0.01:
                idf = 0.01

            for i, tf in self.postings.get(f, []):
                scores[i] += idf * (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * (self.doc_lens[i] / self.avgdl))
                )

        return scores

    def _clean_prediction(self, response: str) -> str:
        if response is None:
            return ""
        prediction = str(response).strip()
        for p in [
            "[Intent]:", "Intent:", "intent:",
            "[Label]:", "Label:", "label:",
            "Answer:", "answer:", "意图:", "分类:"
        ]:
            prediction = prediction.replace(p, "")

        prediction = prediction.strip()
        prediction = prediction.strip("`'\"“”‘’ \t\r\n。；;,.*")
        return prediction

    def _parse_prediction(self, response: str):

        clean_resp = re.sub(r'<analyze>.*?</analyze>', '', str(response), flags=re.DOTALL | re.IGNORECASE)
        clean_resp = re.sub(r'<(think|thought)>.*?</\1>', '', clean_resp, flags=re.DOTALL | re.IGNORECASE)

        prediction = self._clean_prediction(clean_resp)

        if prediction in self.all_labels:
            return prediction

        lower_map = {l.lower(): l for l in self.all_labels}
        if prediction.lower() in lower_map:
            return lower_map[prediction.lower()]

        first_line = prediction.splitlines()[0].strip().strip("`'\"“”‘’ \t\r\n。；;,.*")
        if first_line in self.all_labels:
            return first_line
        if first_line.lower() in lower_map:
            return lower_map[first_line.lower()]

        low = prediction.lower()
        for label in sorted(self.all_labels, key=len, reverse=True):
            if label.lower() in low:
                return label

        return None

    def _make_primary_messages(self, text: str, ranked_indices, scores):
        label_score_sums = collections.defaultdict(float)
        has_positive = False

        for idx in ranked_indices:
            score = scores[int(idx)]
            if score > 0:
                has_positive = True
                label = self.labels[int(idx)]
                label_score_sums[label] += score

        sorted_valid_labels = sorted(self.all_labels, key=lambda l: label_score_sums[l], reverse=True)
        labels_list_str = "\n".join([f"- {l}" for l in sorted_valid_labels])
        top_label = sorted_valid_labels[0] if sorted_valid_labels else None

        sys_msg = {
            "role": "system",
            "content": (
                "You are an elite intent classification AI.\n\n"
                "# Task\n"
                "Classify the user's input into EXACTLY ONE intent from the Valid Intents list.\n\n"
                "# Valid Intents (Ranked by likelihood):\n"
                f"{labels_list_str}\n\n"
                "# Rules\n"
                "1. Focus on the core ACTION (verbs/commands) and ignore specific entity names.\n"
                "2. You MUST briefly think step-by-step inside <analyze>...</analyze> tags to compare the target with the examples.\n"
                "3. After the </analyze> tag, output ONLY the exact intent name on a new line."
            )
        }

        target_block = (
            f"# Target Task\n"
            f"[Text]: {text}\n"
            f"Write your <analyze> reasoning, then output the intent."
        )

        collected_examples = []
        label_quota = collections.Counter()
        used_texts = set()

        for idx in ranked_indices:
            idx = int(idx)
            if has_positive and scores[idx] <= 0:
                continue

            ex_text_raw = self.raw_texts[idx]
            ex_label = self.labels[idx]
            ex_text_norm = self._norm_key(ex_text_raw)
            if ex_text_norm in used_texts:
                continue

            quota_limit = 6 if ex_label == top_label else 3
            if label_quota[ex_label] >= quota_limit:
                continue

            fake_analysis = f"<analyze> Matches the intent of {ex_label}. </analyze>\n"
            ex_str = f"Input: {ex_text_raw}\n{fake_analysis}Intent: {ex_label}\n\n"

            temp_content = "# Reference Examples\n\n" + "".join(collected_examples) + ex_str + target_block
            if self.count_messages_tokens(
                    [sys_msg, {"role": "user", "content": temp_content}]) > self.max_prompt_tokens:
                break

            collected_examples.append(ex_str)
            label_quota[ex_label] += 1
            used_texts.add(ex_text_norm)

        collected_examples.reverse()

        if not collected_examples:
            user_content = target_block
        else:
            user_content = "# Reference Examples\n\n" + "".join(collected_examples) + target_block

        return [sys_msg, {"role": "user", "content": user_content}]

    def predict(self, text: str) -> str:
        if not self.all_labels:
            return ""

        key = self._norm_key(text)
        if key in self.exact_lookup:
            cnt = self.exact_lookup[key]
            return cnt.most_common(1)[0][0]

        query_features = self.extract(text)
        scores = self._get_bm25_scores(query_features)
        ranked_indices = np.argsort(scores)[::-1]

        if len(ranked_indices) > 0 and scores[int(ranked_indices[0])] > 0:
            fallback_label = self.labels[int(ranked_indices[0])]
        else:
            fallback_label = self.global_label_counts.most_common(1)[0][0]

        messages = self._make_primary_messages(text, ranked_indices, scores)
        response = self.call_llm(messages)
        pred = self._parse_prediction(response)

        if pred in self.all_labels:
            return pred

        return fallback_label