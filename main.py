import logging
import chess
from logger_config import setup_logging
from environment import PawnGameEnv
from players import HumanPlayer, SmartPlayer, RandomPlayer, RLAgentPlayer
from gui import PawnGameGUI

# Setup main orchestration logger
logger = logging.getLogger("GameMaster")

def play_game(player1: PawnGameEnv, player2: PawnGameEnv):
    # note: no need those lines with GUI!
    env = PawnGameEnv()
    obs, _ = env.reset()
    env.render()

    current_turn = player1 if player1.color == chess.WHITE else player2
    logger.info(f"Match started! White: {player1.name} vs Black: {player2.name}")

    while True:
        # If no legal actions exist, it's a structural draw or stalemate
        if not env.get_legal_actions():
            logger.info("No legal moves remaining on the board. Structural block/stalemate achieved.")
            print("No legal moves remaining! Game Over.")
            break

        action = current_turn.choose_action(env)
        _, reward, terminated, _, _ = env.step(action)
        env.render()

        if terminated:
            logger.info("Termination signal caught from game rules framework.")
            if reward > 0:
                logger.info(f"Match Finished. Winner: White ({player1.name if player1.color == chess.WHITE else player2.name})")
            elif reward < 0:
                logger.info(f"Match Finished. Winner: Black ({player1.name if player1.color == chess.BLACK else player2.name})")
            else:
                logger.info("Match Finished. Outcome: Draw")
            break

        # Switch turns
        current_turn = player2 if current_turn == player1 else player1


def old_main():
    # Initialize unified logging profile
    setup_logging()

    # Easily hot-swap strategy profiles due to OOP architecture
    # p1 = SmartPlayer(chess.WHITE, "SmartBot-Alpha")
    # p2 = RandomPlayer(chess.BLACK, "RandomBot-Beta")

    # p1 = RLAgentPlayer(chess.WHITE, "RL-SuperBrain", "pawn_game_rl_agent")
    # p2 = HumanPlayer(chess.BLACK, "You")

    # Define your matchup here!
    # P1 (White) is our newly trained RL brain
    p1 = RLAgentPlayer(chess.WHITE, "RL-Bot", "pawn_game_rl_agent")

    # P2 (Black) can be a HumanPlayer ("You"), a SmartPlayer, or a RandomPlayer
    p2 = HumanPlayer(chess.BLACK, "Human-Player")

    play_game(p1, p2)


def main():
    # Initialize unified logging profile
    setup_logging()

    # 1. Create the single environment instance
    env = PawnGameEnv()
    # env.reset()

    # Pass options to explicitly disable scrambling for your match
    env.reset(options={"scramble_board": False})

    # 2. Assign the players
    # white_p = RLAgentPlayer(chess.WHITE, "RL-Bot", "pawn_game_rl_agent")
    white_p = RLAgentPlayer(chess.WHITE, "Self-Play-Bot", "self_play_agent_latest")
    black_p = HumanPlayer(chess.BLACK, "Human")

    # 3. Hand everything off to the GUI
    print("Launching Graphical Interface...")
    PawnGameGUI(env, white_p, black_p)


if __name__ == "__main__":
    main()
