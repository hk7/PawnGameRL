import tkinter as tk
import chess
from environment import PawnGameEnv

class PawnGameGUI:
    def __init__(self, env: PawnGameEnv, white_player, black_player):
        self.env = env
        self.players = {chess.WHITE: white_player, chess.BLACK: black_player}

        self.root = tk.Tk()
        self.root.title("Pawn Game - Reinforcement Learning")

        self.square_size = 70
        self.canvas = tk.Canvas(self.root, width=8 * self.square_size, height=8 * self.square_size)
        self.canvas.pack()

        # High-visibility Unicode chess pieces
        self.piece_symbols = {
            'P': '♙',  # White Pawn
            'p': '♟',  # Black Pawn
            'Q': '♕',  # White Queen
            'q': '♛',  # Black Queen
        }

        self.selected_square = None
        self.canvas.bind("<Button-1>", self.on_square_click)

        self.draw_board()
        self.root.after(500, self.check_ai_turn)
        self.root.mainloop()

    def draw_board(self):
        self.canvas.delete("all")
        colors = ["#f0d9b5", "#b58863"]  # Traditional light/dark wood colors

        for row in range(8):
            for col in range(8):
                # Calculate coordinates
                x1, y1 = col * self.square_size, row * self.square_size
                x2, y2 = x1 + self.square_size, y1 + self.square_size

                # Flip row representation for chess orientation (rank 8 is top, rank 1 is bottom)
                chess_square = chess.square(col, 7 - row)

                # Highlight selected square
                if self.selected_square == chess_square:
                    color = "#7b9c60"
                else:
                    color = colors[(row + col) % 2]

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                # Render pieces
                piece = self.env.board.piece_at(chess_square)
                if piece:
                    symbol = self.piece_symbols.get(piece.symbol(), piece.symbol())
                    # Adjusting text colors dynamically to make them easier to see
                    text_color = "#000000" if piece.color == chess.BLACK else "#ffffff"
                    self.canvas.create_text(
                        x1 + self.square_size/2, y1 + self.square_size/2,
                        text=symbol, font=("Courier", 36, "bold"), fill=text_color
                    )

    def old_on_square_click(self, event):
        current_turn = self.env.board.turn
        # Ignore clicks if it's currently an AI player's turn
        if getattr(self.players[current_turn], "is_ai", False):
            return

        col = event.x // self.square_size
        row = event.y // self.square_size
        clicked_square = chess.square(col, 7 - row)

        if self.selected_square is None:
            # First click: Select a piece belonging to the active player
            piece = self.env.board.piece_at(clicked_square)
            if piece and piece.color == current_turn:
                self.selected_square = clicked_square
                self.draw_board()
        else:
            # Second click: Attempt to move to target square
            move = chess.Move(self.selected_square, clicked_square)
            # Automatic implicit queen promotion matching our underlying environment logic
            if self.env.board.piece_at(self.selected_square) and self.env.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                if chess.square_rank(clicked_square) in [0, 7]:
                    move.promotion = chess.QUEEN

            legal_moves = list(self.env.board.legal_moves)
            if move in legal_moves:
                action_idx = move.from_square * 64 + move.to_square
                _, _, terminated, _, _ = self.env.step(action_idx)
                self.selected_square = None
                self.draw_board()

                if terminated:
                    self.end_game()
                    return

                # Prompt subsequent AI turn calculations
                self.root.after(300, self.check_ai_turn)
            else:
                # Reset if the move target selection was completely invalid
                self.selected_square = None
                self.draw_board()

    def old_check_ai_turn(self):
        if self.env.board.is_game_over() or not list(self.env.board.legal_moves):
            return

        current_turn = self.env.board.turn
        player = self.players[current_turn]

        if getattr(player, "is_ai", False):
            action = player.choose_action(self.env)
            _, _, terminated, _, _ = self.env.step(action)
            self.draw_board()

            if terminated:
                self.end_game()
                return

            # Loop check for fast sequential AI routines if necessary
            self.root.after(300, self.check_ai_turn)

    def old_end_game(self):
        outcome = "Draw Game!"
        if self.env.board.outcome() and self.env.board.outcome().winner == chess.WHITE:
            outcome = "White Wins!"
        elif self.env.board.outcome() and self.env.board.outcome().winner == chess.BLACK:
            outcome = "Black Wins!"

        popup = tk.Toplevel(self.root)
        popup.title("Game Over")
        tk.Label(popup, text=outcome, font=("Helvetica", 16, "bold"), padx=20, pady=20).pack()
        tk.Button(popup, text="Close", command=self.root.quit).pack(pady=10)


    def on_square_click(self, event):
        current_turn = self.env.board.turn
        if getattr(self.players[current_turn], "is_ai", False):
            return

        col = event.x // self.square_size
        row = event.y // self.square_size
        clicked_square = chess.square(col, 7 - row)

        if self.selected_square is None:
            piece = self.env.board.piece_at(clicked_square)
            if piece and piece.color == current_turn:
                self.selected_square = clicked_square
                self.draw_board()
        else:
            move = chess.Move(self.selected_square, clicked_square)
            if self.env.board.piece_at(self.selected_square) and self.env.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                if chess.square_rank(clicked_square) in [0, 7]:
                    move.promotion = chess.QUEEN

            legal_moves = list(self.env.board.legal_moves)
            if move in legal_moves:
                action_idx = move.from_square * 64 + move.to_square

                # --- FIX: Capture the reward variable here ---
                _, reward, terminated, _, _ = self.env.step(action_idx)

                self.selected_square = None
                self.draw_board()

                if terminated:
                    self.end_game(reward)  # Pass reward to end_game
                    return

                self.root.after(300, self.check_ai_turn)
            else:
                self.selected_square = None
                self.draw_board()

    def check_ai_turn(self):
        if self.env.board.is_game_over() or not list(self.env.board.legal_moves):
            return

        current_turn = self.env.board.turn
        player = self.players[current_turn]

        if getattr(player, "is_ai", False):
            # --- FIX: Capture the reward variable here ---
            action = player.choose_action(self.env)
            _, reward, terminated, _, _ = self.env.step(action)
            self.draw_board()

            if terminated:
                self.end_game(reward)  # Pass reward to end_game
                return

            self.root.after(300, self.check_ai_turn)

    def end_game(self, final_reward):
        # --- FIX: Read the reward configuration directly ---
        if final_reward > 0:
            outcome = "White Wins!"
        elif final_reward < 0:
            outcome = "Black Wins!"
        else:
            outcome = "Draw Game!"

        popup = tk.Toplevel(self.root)
        popup.title("Game Over")
        tk.Label(popup, text=outcome, font=("Helvetica", 16, "bold"), padx=20, pady=20).pack()
        tk.Button(popup, text="Close", command=self.root.quit).pack(pady=10)
