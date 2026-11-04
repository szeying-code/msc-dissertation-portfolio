"""Command line menu (script entry point)."""

from .robustness import run_degree_robustness_experiment
from .workflow import reconstruct_field, train_model


def main():
    """Script entry point for user interaction."""
    best_grad = None
    while True:
        menu = input('(1) Train model, (2) Reconstruct data, (3) Degree robustness experiment, (4) Exit: ')
        if menu == '1':
            model, best_grad = train_model()
            print(
                f"\nBest gradient available for degree robustness: "
                f"{best_grad}"
            )
        elif menu == '2':
            n = input('Enter the number of points to use to create an n^2 sized grid for field reconstruction: ')
            if int(n) <= 0:
                print('Invalid choice. n cannot be less than or equal to 0.')
                continue
            reconstruct_field(int(n))
        elif menu == '3':
            if best_grad is None:
                print(
                    "\nPlease train the model first "
                    "before running the degree robustness experiment."
                )
                continue
            print(
                f"\nRunning degree robustness using best grad = "
                f"{best_grad}"
            )
            run_degree_robustness_experiment(best_grad)
        elif menu == '4':
            break


    
if __name__ == "__main__":
    main()
