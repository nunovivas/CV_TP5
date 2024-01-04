class DifficultyHandler:
    def __init__(self, initial_difficulty='easy'):
        self.current_difficulty = initial_difficulty

    def loop_difficulty(self):
        print (F"Current difficulty:{self.current_difficulty}")
        match self.current_difficulty.lower():
            case 'easy':
                self.current_difficulty='Medium'
            case 'medium':
                self.current_difficulty='Hard'
            case 'hard':
                self.current_difficulty='God'
            case 'god':
                self.current_difficulty='Easy'
            case _:
                self.current_difficulty='easy' # so it's somewhat visible that something weird occurred
        return self.current_difficulty

    def get_current_difficulty(self):
        return self.current_difficulty