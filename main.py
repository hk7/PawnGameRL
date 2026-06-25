import logging
import chess
from logger_config import setup_logging
from environment import PawnGameEnv
from players import HumanPlayer, SmartPlayer, RandomPlayer

# Setup main orchestration logger
logger = logging.getLogger("GameMaster")

def play_game(player1: PawnGameEnv, player2: PawnGameEnv):
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


def main():
    # Initialize unified logging profile
    setup_logging()
    
    # Easily hot-swap strategy profiles due to OOP architecture
    p1 = SmartPlayer(chess.WHITE, "SmartBot-Alpha")
    p2 = RandomPlayer(chess.BLACK, "RandomBot-Beta")
    
    play_game(p1, p2)

if __name__ == "__main__":
    main()
    