def viterbi(obs, states, start_p, trans_p, emit_p):
    # 初始化：记录每个状态的最大概率和路径来源
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    
    # 递推：计算每一天每个状态的最大概率
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(
                V[t-1][prev_st]["prob"] * trans_p[prev_st][st] 
                for prev_st in states
            )
            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break
    
    # 回溯：找到最优路径
    path = []
    max_prob = max(value["prob"] for value in V[-1].values())
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            path.append(st)
            break
    
    for t in range(len(V)-2, -1, -1):
        path.insert(0, V[t+1][path[0]]["prev"])
    
    return path, [v for v in V]
if __name__ == "__main__":
    # 定义所有参数（用字典更直观）
    states = ["Sunny", "Rainy"]
    obs = ["购物", "打扫", "打游戏"]  # 对应索引0,1,2

    # 初始概率（索引对应states）
    start_p = {"Sunny": 0.6, "Rainy": 0.4}

    # 转移概率（from行 to列）
    trans_p = {
        "Sunny": {"Sunny": 0.7, "Rainy": 0.3},
        "Rainy": {"Sunny": 0.4, "Rainy": 0.6}
    }

    # 发射概率（天气状态下行为的概率）
    emit_p = {
        "Sunny": [0.3, 0.5, 0.2],  # 购物(0), 打扫(1), 打游戏(2)
        "Rainy": [0.1, 0.3, 0.6]
    }

    # 运行维特比算法
    path, V = viterbi(
        obs=[0, 1, 2],  # 购物(0), 打扫(1), 打游戏(2)
        states=states,
        start_p=start_p,
        trans_p=trans_p,
        emit_p=emit_p
    )

    print("最优天气序列：", path)
    print("详细概率变化：")
    for t in range(len(V)):
        print(f"Day {t+1}:")
        for st in states:
            print(f"  {st}: {V[t][st]['prob']:.5f}")
            
            
    """
    一、前置知识：马尔可夫模型
想象你每天观察天气，但天气预报员是个懒鬼，他只会根据今天的天气预测明天的天气，这就是马尔可夫假设——下一个状态只依赖当前状态。

举个栗子 🌰：

天气有两种状态：晴天☀️（Sunny）和雨天🌧️（Rainy）
转移概率：
今天晴天→明天晴天：70%
今天晴天→明天雨天：30%
今天雨天→明天晴天：40%
今天雨天→明天雨天：60%
用矩阵表示：

# 行是当前状态，列是下一状态
transition = [
    [0.7, 0.3],  # Sunny → [Sunny, Rainy]
    [0.4, 0.6]   # Rainy → [Sunny, Rainy]
]
二、隐马尔可夫模型（HMM）
现在升级难度！假设你被关在地下室，只能通过室友的行为（购物、打扫、打游戏）来猜测外面的天气（晴天/雨天）。这就是HMM：

隐藏状态（天气）：你无法直接观察
观测状态（行为）：你可以直接看到
发射概率：不同天气下室友行为的概率
举个栗子 🌰：

晴天时室友行为概率：
购物：30%
打扫：50%
打游戏：20%
雨天时室友行为概率：
购物：10%
打扫：30%
打游戏：60%
用矩阵表示：

# 行是天气状态，列是行为
emission = [
    [0.3, 0.5, 0.2],  # Sunny → [购物, 打扫, 打游戏]
    [0.1, 0.3, 0.6]   # Rainy → [购物, 打扫, 打游戏]
]
三、维特比算法要解决什么问题？
假设你连续三天观察到室友的行为是：[购物, 打扫, 打游戏]，想知道这三天最可能的天气序列是什么？

暴力解法：列出所有可能的天气组合（2^3=8种），计算每种的概率，选最大的。但状态数指数增长，效率太低！

维特比算法：用动态规划，时间复杂度从指数级降到线性级！

维特比算法核心思想:
递推计算：每天记录到达每个天气的最大概率路径
回溯路径：从最后一天倒推，找到最优路径


手把手计算过程
已知条件
初始概率：晴天60%，雨天40%
转移概率和发射概率见上文
观测序列：购物（第1天）、打扫（第2天）、打游戏（第3天）
第1天：购物
计算初始概率 × 发射概率：
晴天：0.6 × 0.3 = 0.18
雨天：0.4 × 0.1 = 0.04
天气	概率	前一天的天气
☀️	0.18	无（第1天）
🌧️	0.04	无（第1天）
第2天：打扫
对每个天气，找前一天的最大概率路径：

今天晴天☀️：

昨天晴天→今天晴天：0.18 × 0.7 × 0.5 = 0.063
昨天下雨→今天晴天：0.04 × 0.4 × 0.5 = 0.008
最大概率：0.063（来自昨天晴天）
今天下雨🌧️：

昨天晴天→今天下雨：0.18 × 0.3 × 0.3 = 0.0162
昨天下雨→今天下雨：0.04 × 0.6 × 0.3 = 0.0072
最大概率：0.0162（来自昨天晴天）
天气	概率	前一天的天气
☀️	0.063	☀️
🌧️	0.0162	☀️
第3天：打游戏
继续同样逻辑：

今天晴天☀️：

昨天晴天→今天晴天：0.063 × 0.7 × 0.2 = 0.00882
昨天下雨→今天晴天：0.0162 × 0.4 × 0.2 = 0.001296
最大概率：0.00882（来自昨天晴天）
今天下雨🌧️：

昨天晴天→今天下雨：0.063 × 0.3 × 0.6 = 0.01134
昨天下雨→今天下雨：0.0162 × 0.6 × 0.6 = 0.005832
最大概率：0.01134（来自昨天晴天）
天气	概率	前一天的天气
☀️	0.00882	☀️
🌧️	0.01134	☀️
回溯路径
第3天最大概率是雨天🌧️（0.01134），来自第2天晴天☀️
第2天晴天来自第1天晴天☀️
最终路径：☀️ → ☀️ → 🌧️
    
    
    
    """
def viterbi_speech(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    # 初始化
    for st in states:
        if start_p[st] > 0:  # 只考虑起始状态m和g
            V[0][st] = {
                "prob": start_p[st] * emit_p[st][obs[0]],
                "path": [st]
            }
    
    # 递推
    for t in range(1, len(obs)):
        V.append({})
        for curr_st in states:
            max_prob = -1
            best_prev_st = None
            for prev_st in states:
                if prev_st in V[t-1] and trans_p[prev_st].get(curr_st, 0) > 0:
                    prob = V[t-1][prev_st]["prob"] * trans_p[prev_st][curr_st] * emit_p[curr_st][obs[t]]
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_st = prev_st
            if best_prev_st is not None:
                V[t][curr_st] = {
                    "prob": max_prob,
                    "path": V[t-1][best_prev_st]["path"] + [curr_st]
                }
    
    # 找出最大概率路径
    max_path = None
    max_prob = 0
    for st in states:
        if st in V[-1] and V[-1][st]["prob"] > max_prob:
            max_prob = V[-1][st]["prob"]
            max_path = V[-1][st]["path"]
    
    return max_path, max_prob
# 隐藏状态：m, a, o, g, u
states = ["m", "a", "o", "g", "u"]

# 初始概率（假设语音开头更可能是辅音）
start_p = {
    "m": 0.5,  # "猫"的起始音素
    "g": 0.5,  # "狗"的起始音素
    "a": 0.0, "o": 0.0, "u": 0.0
}

# 转移概率（音素之间的连接规律）
trans_p = {
    "m": {"a": 1.0, "o": 0.0, "g": 0.0, "u": 0.0},  # m只能转a
    "a": {"o": 1.0, "m": 0.0, "g": 0.0, "u": 0.0},  # a只能转o
    "o": {"u": 0.0, "m": 0.0, "g": 0.0, "a": 0.0},  # o是终点（无转移）
    "g": {"o": 1.0, "m": 0.0, "a": 0.0, "u": 0.0},  # g只能转o
    "u": {"o": 0.0, "m": 0.0, "g": 0.0, "a": 0.0}   # u是终点
}

# 发射概率（音素产生音高的概率）
emit_p = {
    "m": [0.7, 0.2, 0.1],  # m音素：低音(70%)，中音(20%)，高音(10%)
    "a": [0.1, 0.8, 0.1],  # a音素：中音为主
    "o": [0.2, 0.3, 0.5],  # o音素：高音较多
    "g": [0.6, 0.3, 0.1],  # g音素：低音为主
    "u": [0.1, 0.1, 0.8]   # u音素：高音为主
}

# 观测序列：假设检测到音高为 [低, 中, 高]
obs = ["L", "M", "H"]  # 对应索引0,1,2

# 将观测值转换为索引（L:0, M:1, H:2）
obs_indices = [0, 1, 2]
path, prob = viterbi_speech(
    obs=obs_indices,
    states=states,
    start_p=start_p,
    trans_p=trans_p,
    emit_p=emit_p
)

print("最优音素路径:", path)
print("对应单词:", "猫" if path[-1] == "o" else "狗")
print("路径概率:", prob)

"""
第一步：建立隐马尔可夫模型（HMM）
假设每个单词由3个音素组成：

"猫"（mao）：音素序列 m → a → o
"狗"（gou）：音素序列 g → o → u
每个音素对应一个隐藏状态，观测值是声音信号的简化特征（这里用音高表示）：

低音（L）
中音（M）
高音（H）
    
    第三步：维特比算法计算过程
假设观测序列为 [低, 中, 高]，要判断是"猫"还是"狗"。

Day 1（低音）：

只可能从起始状态m或g出发
m的概率：0.5 × 0.7 = 0.35
g的概率：0.5 × 0.6 = 0.30
Day 2（中音）：

当前状态只能是a或o（由转移规则决定）
a：必须来自m → 0.35 × 1.0 × 0.8 = 0.28
o：可能来自g → 0.30 × 1.0 × 0.5 = 0.15
Day 3（高音）：

当前状态只能是o或u
o：来自a → 0.28 × 1.0 × 0.5 = 0.14
u：来自o（但o无法转u）→ 无效路径
u：来自g的路径已中断
最优路径：m → a → o → 对应单词"猫"
    
    
"""