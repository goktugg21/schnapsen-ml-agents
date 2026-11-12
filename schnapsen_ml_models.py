from schnapsen.game import Bot, PlayerPerspective, Move, SchnapsenGamePlayEngine, Score, GamePhase
from schnapsen.deck import Card, Rank, Suit
from schnapsen.bots import RandBot, BullyBot, RdeepBot
from schnapsen.bots.ml_bot import MLDataBot, MLPlayingBot, MLPClassifier, train_ML_model
import pathlib
import random
import time
import shutil
from typing import Optional, cast, List, Dict, Tuple
import traceback
from schnapsen.game import SchnapsenTrickScorer
import os

class Phase2FocusedBot(Bot):
    """Bot focused on Phase 2 excellence while maintaining simple Phase 1 strategy."""
    
    def __init__(self, rand: random.Random)-> None:
        super().__init__()
        self.__random: random.Random = rand
    
    def _get_move_card(self, move: Move)-> Optional[Card]:
        """Extract card from any move type safely - returns queen for marriages and jack for exchanges."""
        if move.is_marriage():
            return move.queen_card  
        elif move.is_trump_exchange():
            return move.jack
        elif hasattr(move, 'card'):
            return move.card
        return None
        
    def _get_card_points(self, card: Card)-> int:
        """Calculate card points according to Schnapsen scoring rules."""
        points: Dict[Rank, int] = {
            Rank.ACE: 11,
            Rank.TEN: 10,
            Rank.KING: 4,
            Rank.QUEEN: 3,
            Rank.JACK: 2
        }
        return points.get(card.rank, 0)
    
    def _evaluate_phase2_position(self, perspective: PlayerPerspective)-> Dict[str, int]:
        """Analyze current game state for Phase 2 strategy decisions."""
        hand = perspective.get_hand()
        trump_suit = perspective.get_trump_suit()
        score = perspective.get_my_score()
        
        # Count trump and high-value cards in hand.
        trump_count: int = 0
        high_card_count: int = 0
        hand_points: int = 0
        
        for card in hand.cards:
            if card.suit == trump_suit:
                trump_count += 1
            if self._get_card_points(card) >= 10:
                high_card_count += 1
            hand_points += self._get_card_points(card)
        
        metrics: Dict[str, int] = {
            'trump_cards': trump_count,
            'high_cards': high_card_count,
            'total_points': score.direct_points + score.pending_points,
            'hand_points': hand_points
        }
        return metrics
    
    def get_move(self, perspective: PlayerPerspective, leader_move: Optional[Move])-> Move:
        """
        Determine and return the best move based on game phase and current state.
   
        Args:
            perspective (PlayerPerspective): Current game state and player's view.
            leader_move (Optional[Move]): Move made by leading player if following, None if leading.
   
        Returns:
            Move: Selected move to play.
        """
        valid_moves: List[Move] = perspective.valid_moves()
        trump_suit: Suit = perspective.get_trump_suit()
        
         # Phase 1 strategy - Focus on marriages and trump exchanges.
        if perspective.get_phase().value == 1:
            # Check for marriage opportunities, prioritizing trump marriages.
            marriage_moves: List[Move] = [marriage_move for marriage_move in valid_moves if marriage_move.is_marriage()]
            if marriage_moves:
                trump_marriages: List[Move] = [trump_marriage for trump_marriage in marriage_moves if trump_marriage.queen_card.suit == trump_suit]
                return self.__random.choice(trump_marriages) if trump_marriages else self.__random.choice(marriage_moves)
       
            # Take trump exchange if available.
            exchange_moves: List[Move] = [exchange for exchange in valid_moves if exchange.is_trump_exchange()]
            if exchange_moves:
                return exchange_moves[0]
           
            return self.__random.choice(valid_moves)
        
        # Phase 2 strategy - Focus on point maximization.
        metrics: Dict[str, int] = self._evaluate_phase2_position(perspective)
   
        # Leading player strategy.
        if leader_move is None:
            # If holding high cards, try to play them.
            if metrics['high_cards'] > 0:
                regular_moves: List[Move] = []
                for potential_move in valid_moves:
                    if not potential_move.is_marriage() and not potential_move.is_trump_exchange():
                        regular_moves.append(potential_move)
                 
                if regular_moves:
                    # Calculate points for each move.
                    move_points: List[Tuple[Move, int]] = []
                    for current_move in regular_moves:
                        card = self._get_move_card(current_move)
                        if card:
                            points = self._get_card_points(card)
                            move_points.append((current_move, points))
               
                    # Select from moves worth 10+ points.
                    best_moves: List[Move] = [move for move, points in move_points if points >= 10]
                    if best_moves:
                        return self.__random.choice(best_moves)
            
            # Try to play lowest non-trump card.
            non_trump_moves: List[Move] = []
            for current_move in valid_moves:
                if (not current_move.is_marriage() and not current_move.is_trump_exchange() and 
                    self._get_move_card(current_move) and self._get_move_card(current_move).suit != trump_suit):
                    non_trump_moves.append(current_move)
       
            if non_trump_moves:
                return min(non_trump_moves, key=lambda move: self._get_card_points(self._get_move_card(move)))
   
        # Following player strategy.
        else:
            leader_card: Optional[Card] = self._get_move_card(leader_move)
            if not leader_card:
                return self.__random.choice(valid_moves)
       
            # Try to follow suit.
            following_moves: List[Move] = []
            for follow_move in valid_moves:
                move_card = self._get_move_card(follow_move)
                if (not follow_move.is_marriage() and not follow_move.is_trump_exchange() and 
                    move_card and move_card.suit == leader_card.suit):
                    following_moves.append(follow_move)
       
            if following_moves:
                # For valuable tricks, try to win with lowest winning card.
                if self._get_card_points(leader_card) >= 10:
                    winning_moves: List[Move] = []
                    for current_move in following_moves:
                        move_card = self._get_move_card(current_move)
                        if move_card and self._get_card_points(move_card) > self._get_card_points(leader_card):
                            winning_moves.append(current_move)
               
                    if winning_moves:
                        return min(winning_moves, key=lambda move: self._get_card_points(self._get_move_card(move)))
           
                # Play lowest following card if can't/shouldn't win.
                return min(following_moves, key=lambda move: self._get_card_points(self._get_move_card(move)))
       
            # For valuable tricks, consider trumping.
            if self._get_card_points(leader_card) >= 10:
                trump_moves: List[Move] = []
                for trump_move in valid_moves:
                    move_card = self._get_move_card(trump_move)
                    if (not trump_move.is_marriage() and not trump_move.is_trump_exchange() and 
                        move_card and move_card.suit == trump_suit):
                        trump_moves.append(trump_move)
           
                if trump_moves:
                    return min(trump_moves, key=lambda move: self._get_card_points(self._get_move_card(move)))
   
        # Default to random move if no strategy applies.
        return self.__random.choice(valid_moves)

def create_training_data(num_games: int = 10000)-> List[MLDataBot]:
    """Generate training data by playing games between different bot types."""
    # Clean up and create necessary directories.
    for directory in ['ML_replay_memories', 'ML_models']:
        path: pathlib.Path = pathlib.Path(directory)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    
    # Initialize different bot types for diverse training.
    phase2_bot: Phase2FocusedBot = Phase2FocusedBot(random.Random(42))
    rdeep_bot: RdeepBot = RdeepBot(num_samples=12, depth=4, rand=random.Random(43))
    bully_bot: BullyBot = BullyBot(random.Random(44))
    
    # Create data collection bots.
    data_bots: List[MLDataBot] = [
        MLDataBot(bot=phase2_bot, replay_memory_location=pathlib.Path('ML_replay_memories/training.txt')),
        MLDataBot(bot=rdeep_bot, replay_memory_location=pathlib.Path('ML_replay_memories/training.txt')),
        MLDataBot(bot=bully_bot, replay_memory_location=pathlib.Path('ML_replay_memories/training.txt'))
    ]
    
    engine: SchnapsenGamePlayEngine = SchnapsenGamePlayEngine()
    print(f"Playing {num_games} games to collect training data...")
    
    # Play games to collect training data.
    for game_index in range(num_games):
        if game_index % 100 == 0:
            print(f"Playing game {game_index}/{num_games}")
        
        selected_bots: Tuple[MLDataBot, MLDataBot] = random.sample(data_bots, 2)
        try:
            engine.play_game(selected_bots[0], selected_bots[1], random.Random(game_index))
        except Exception as error:
            print(f"Error in game {game_index}: {error}")
            continue
    
    return data_bots

def train_models()-> None:
    """Train both neural network and logistic regression models."""
    try:
        print("Training Neural Network...")
        train_ML_model(
            replay_memory_location=pathlib.Path('ML_replay_memories/training.txt'),
            model_location=pathlib.Path('ML_models/neural_net_model'),
            model_class='NN'
        )
        
        print("\nTraining Logistic Regression...")
        train_ML_model(
            replay_memory_location=pathlib.Path('ML_replay_memories/training.txt'),
            model_location=pathlib.Path('ML_models/logistic_reg_model'),
            model_class='LR'
        )
        
    except Exception as error:
        print(f"Error during training: {error}")

def play_test_games(bot1: Bot, bot2: Bot, num_games: int = 100)-> Dict[str, float]:
    """Play test games and collect performance statistics."""
    engine: SchnapsenGamePlayEngine = SchnapsenGamePlayEngine()
    stats: Dict[str, float] = {
        'wins': 0,
        'total_time': 0,
        'phase_one_wins': 0,
        'phase_two_wins': 0,
        'total_points': 0,
        'games_played': 0
    }
    
    # Play test games and collect statistics.
    for game_index in range(num_games):
        try:
            start_time: float = time.time()
            winner, points, score = engine.play_game(bot1, bot2, random.Random(game_index))
            
            stats['games_played'] += 1
            stats['total_time'] += time.time() - start_time
            
            # Track win statistics.
            if isinstance(winner, type(bot1)) and id(winner) == id(bot1):
                stats['wins'] += 1
                stats['total_points'] += score.direct_points
                if score.direct_points < 66:
                    stats['phase_one_wins'] += 1
                else:
                    stats['phase_two_wins'] += 1
            
        except Exception as error:
            print(f"Error in game {game_index}: {error}")
            continue
    
    # Calculate performance metrics.
    if stats['games_played'] > 0:
        stats.update({
            'win_rate': (stats['wins'] / stats['games_played']) * 100,
            'avg_time': stats['total_time'] / stats['games_played'],
            'avg_points': stats['total_points'] / max(stats['wins'], 1),
            'phase_one_win_rate': (stats['phase_one_wins'] / stats['games_played']) * 100,
            'phase_two_win_rate': (stats['phase_two_wins'] / stats['games_played']) * 100
        })
    
    return stats

def test_models(num_games: int = 200)-> Dict[str, Dict[str, float]]:
    """Test both models against various opponents and each other."""
    try:
        # Initialize models and opponents.
        nn_bot: MLPlayingBot = MLPlayingBot(model_location=pathlib.Path('ML_models/neural_net_model'))
        lr_bot: MLPlayingBot = MLPlayingBot(model_location=pathlib.Path('ML_models/logistic_reg_model'))
        phase2_bot: Phase2FocusedBot = Phase2FocusedBot(random.Random(42))
        
        opponents: Dict[str, Bot] = {
            'Random': RandBot(random.Random(43)),
            'Bully': BullyBot(random.Random(44)),
            'Phase2': phase2_bot,
            'Rdeep-Basic': RdeepBot(num_samples=8, depth=3, rand=random.Random(45)),
            'Rdeep-Advanced': RdeepBot(num_samples=12, depth=4, rand=random.Random(46))
        }
        
        results: Dict[str, Dict[str, float]] = {}
        
        # Test both ML models against all opponents.
        for model_name, model_bot in [('Neural Network', nn_bot), ('Logistic Regression', lr_bot)]:
            print(f"\nTesting {model_name}...")
            
            for opp_name, opponent in opponents.items():
                print(f"\nPlaying against {opp_name}")
                
                # Test from both positions.
                print("Testing as first player...")
                stats_first: Dict[str, float] = play_test_games(model_bot, opponent, num_games // 2)
                print("Testing as second player...")
                stats_second: Dict[str, float] = play_test_games(opponent, model_bot, num_games // 2)
                
                # Combine statistics from both positions.
                total_wins: int = int(stats_first['wins'] + (num_games // 2 - stats_second['wins']))
                phase_one_wins: int = int(stats_first['phase_one_wins'] + stats_second['phase_one_wins'])
                phase_two_wins: int = int(stats_first['phase_two_wins'] + stats_second['phase_two_wins'])
                
                combined_stats: Dict[str, float] = {
                    'wins': float(total_wins),
                    'games_played': float(num_games),
                    'phase_one_wins': float(phase_one_wins),
                    'phase_two_wins': float(phase_two_wins),
                    'win_rate': (total_wins / num_games) * 100,
                    'phase_one_win_rate': (phase_one_wins / num_games) * 100,
                    'phase_two_win_rate': (phase_two_wins / num_games) * 100
                }
                
                results[f'{model_name} vs {opp_name}'] = combined_stats
                
                # Display intermediate results.
                print(f"Win rate: {combined_stats['win_rate']:.1f}%")
                print(f"Phase 1 win rate: {combined_stats['phase_one_win_rate']:.1f}%")
                print(f"Phase 2 win rate: {combined_stats['phase_two_win_rate']:.1f}%")
        
        # Head-to-head model comparison.
        print("\nTesting Models Head-to-Head")
        head_to_head_stats: Dict[str, float] = play_test_games(nn_bot, lr_bot, num_games)
        results['Model Competition'] = head_to_head_stats
        
        return results
        
    except Exception as error:
        print(f"Error during testing: {error}")
        traceback.print_exc()
        return {}
    
if __name__ == "__main__":
    # Execute training and testing pipeline.
    print("Step 1: Creating training data...")
    data_bots: List[MLDataBot] = create_training_data(num_games=10000)
    
    print("\nStep 2: Training models...")
    train_models()
    
    print("\nStep 3: Testing models...")
    results: Dict[str, Dict[str, float]] = test_models(num_games=200)
    
    # Display final test results.
    print("\nFinal Results:")
    print("-" * 50)
    for match, stats in results.items():
        print(f"{match}:")
        print(f"  Wins: {stats['wins']}/{stats['games_played']} ({stats['win_rate']:.1f}%)")
        print(f"  Phase 1 wins: {stats['phase_one_wins']} ({stats['phase_one_win_rate']:.1f}%)")
        print(f"  Phase 2 wins: {stats['phase_two_wins']} ({stats['phase_two_win_rate']:.1f}%)")
        print("-" * 50)