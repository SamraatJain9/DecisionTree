import speech_recognition as sr
from decision_trees import train_decision_tree

def get_task_type():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        print("Please say 'classification', 'regression', or 'exit' to stop.")
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)
            
        try:
            task_type = recognizer.recognize_google(audio).lower()
            print(f"You said: {task_type}")
            
            if task_type in ['classification', 'regression']:
                return task_type
            elif task_type == 'exit':
                print("Exiting the program.")
                exit()  # Exit the program if the user says 'exit'
            else:
                print("Invalid input. Please say 'classification', 'regression', or 'exit'.")
        
        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said. Please try again.")
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            break

def main():
    while True:  # Keep running the program until the user says "exit"
        task_type = get_task_type()  # Get the task type
        if task_type == 'exit':  # If 'exit' is received, exit the loop
            break
        
        print(f"Running {task_type} task...")  # Print the current task type
        clf_best = train_decision_tree(task_type)  # Train the decision tree with the given task type

if __name__ == '__main__':
    main()
