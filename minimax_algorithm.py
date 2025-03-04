import numpy as np

# 井字棋棋盘初始化
def print_board(board):
    """打印棋盘"""
    for row in board:
        print(" | ".join(row))
    print("-" * 9)

# 检查胜负
def check_winner(board):
    """判断游戏是否结束并返回赢家 ('X', 'O' 或 None)"""
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != " ":
            return row[0]
    
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != " ":
            return board[0][col]
    
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != " ":
        return board[0][0]
    
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != " ":
        return board[0][2]
    
    return None if any(" " in row for row in board) else "Tie"

# 评分函数
def evaluate(board):
    """返回当前棋盘状态的评分"""
    winner = check_winner(board)
    if winner == "X":  # AI 胜利
        return 1
    elif winner == "O":  # 玩家胜利
        return -1
    else:  # 平局
        return 0

# Minimax 递归搜索
def minimax(board, depth, is_max):
    """Minimax 算法"""
    score = evaluate(board)

    # 如果游戏结束（胜利/失败/平局），返回分数
    if score == 1 or score == -1 or score == 0:
        return score

    if is_max:  # AI（X）的回合，选择最大值
        best = -np.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"
                    best = max(best, minimax(board, depth + 1, False))
                    board[i][j] = " "  # 还原
        return best
    else:  # 玩家（O）的回合，选择最小值
        best = np.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"
                    best = min(best, minimax(board, depth + 1, True))
                    board[i][j] = " "  # 还原
        return best

# AI 选择最佳行动
def find_best_move(board):
    """AI 使用 Minimax 选择最佳位置"""
    best_val = -np.inf
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == " ":
                board[i][j] = "X"
                move_val = minimax(board, 0, False)
                board[i][j] = " "  # 还原

                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move
def minimax_alpha_beta(board, depth, is_max, alpha, beta):
    """Minimax + Alpha-Beta 剪枝"""
    score = evaluate(board)

    if score == 1 or score == -1 or score == 0:
        return score

    if is_max:  # AI (X)
        best = -np.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "X"
                    best = max(best, minimax_alpha_beta(board, depth + 1, False, alpha, beta))
                    board[i][j] = " "
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break  # 剪枝
        return best
    else:  # 玩家 (O)
        best = np.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    board[i][j] = "O"
                    best = min(best, minimax_alpha_beta(board, depth + 1, True, alpha, beta))
                    board[i][j] = " "
                    beta = min(beta, best)
                    if beta <= alpha:
                        break  # 剪枝
        return best


# 运行示例
board = [
    ["X", "O", "X"],
    [" ", "O", " "],
    [" ", " ", " "]
]

print("当前棋盘状态:")
print_board(board)

best_move = find_best_move(board)
print(f"AI 选择的位置: {best_move}")
