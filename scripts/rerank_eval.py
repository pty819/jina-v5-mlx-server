#!/usr/bin env python3
"""End-to-end rerank evaluation against the running gateway.

Builds several Chinese/English mixed retrieval cases, POSTs them to /v1/rerank,
and checks ranking quality + response shape.
"""
import json
import sys
import time
import urllib.request
import urllib.error

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.1.238:8000"

# Each case: (label, query, [documents...], expected_top1_index)
# Designed to exercise: multilingual, semantic vs lexical, distractors.
CASES = [
    (
        "en-science",
        "What causes the northern lights?",
        [
            "Auroras are caused by solar wind particles colliding with the Earth's magnetosphere.",
            "The Eiffel Tower is located in Paris, France.",
            "Photosynthesis converts sunlight into chemical energy in plants.",
            "Northern lights, or aurora borealis, appear near the polar regions.",
        ],
        0,
    ),
    (
        "zh-medical",
        "糖尿病的早期症状有哪些?",
        [
            "糖尿病的典型症状包括多饮、多尿、多食和体重下降,即三多一少。",
            "高血压患者应减少盐的摄入量。",
            "感冒通常由病毒引起,症状有流涕和发热。",
            "糖尿病患者早期可能没有明显症状,需通过血糖检测发现。",
        ],
        0,
    ),
    (
        "semantic-not-lexical",
        "How do I make my Python code run faster?",
        [
            "You can use PyPy or Cython to compile Python to native code for speed.",
            "Python is a popular programming language created by Guido van Rossum.",
            "Java requires a JVM to execute bytecode.",
            "The fastest land animal is the cheetah.",
        ],
        0,
    ),
    (
        "distractor-heavy",
        "MLX 框架适用于什么硬件?",
        [
            "MLX 是苹果为 Apple Silicon 设计的机器学习数组框架。",
            "TensorFlow 广泛用于 NVIDIA GPU 上的深度学习训练。",
            "PyTorch 是 Meta 开源的深度学习框架。",
            "CoreML 是苹果用于在设备上部署模型的高层 API。",
        ],
        0,
    ),
    (
        "long-context",
        "什么是分布式系统中的一致性?",
        [
            "一致性是指在分布式系统的多个副本之间保持数据同步的属性,常见模型有强一致性、最终一致性和因果一致性。",
            "数据库事务具有 ACID 特性:原子性、一致性、隔离性、持久性。这里的 consistency 指事务前后数据约束不被破坏。",
            "Raft 和 Paxos 是实现分布式共识的经典算法,用于让多个节点就某个值达成一致。",
            "分布式锁用于在集群中互斥访问共享资源。",
        ],
        0,
    ),
]


def post_rerank(query, documents, top_n=None):
    payload = {
        "model": "jinaai/jina-reranker-v3-mlx",
        "query": query,
        "documents": documents,
        "return_documents": True,
    }
    if top_n is not None:
        payload["top_n"] = top_n
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/v1/rerank",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print(f"gateway: {BASE}")
    passed = 0
    failed = 0
    for label, query, docs, expected_idx in CASES:
        t0 = time.time()
        try:
            body = post_rerank(query, docs)
        except urllib.error.HTTPError as e:
            print(f"[FAIL] {label}: HTTP {e.code} {e.read().decode('utf-8', 'ignore')[:200]}")
            failed += 1
            continue
        except Exception as e:
            print(f"[FAIL] {label}: {type(e).__name__}: {e}")
            failed += 1
            continue
        dt = (time.time() - t0) * 1000
        results = body["results"]
        top = results[0]
        ok = top["index"] == expected_idx
        passed += ok
        failed += not ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {label} ({dt:.0f}ms, tokens={body['usage']['total_tokens']})")
        print(f"       query: {query}")
        for r in results[:3]:
            flag = " <==" if r["index"] == expected_idx else ""
            print(f"       idx={r['index']} score={r['relevance_score']:+.4f} emb={r.get('embedding')} doc={r['document'][:50]!r}{flag}")
        # shape checks
        assert body["model"] == "jinaai/jina-reranker-v3-mlx", f"unexpected model: {body['model']}"
        assert all(r.get("embedding") is None for r in results), "embedding must be null"
        assert all(-1.0 <= r["relevance_score"] <= 1.0 for r in results), "score out of [-1,1]"
    print(f"\nresult: {passed}/{len(CASES)} passed, {failed} failed")


if __name__ == "__main__":
    main()
