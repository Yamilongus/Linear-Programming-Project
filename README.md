# Linear-Programming-Project-SICI4028

Purpose:

The code implements the Simplex algorithm for solving linear programming (LP) problems.
It takes user input for the objective function, constraints, and their types (MAX or MIN).
It then iteratively applies the Simplex method to find the optimal solution or indicate infeasibility.
Key Components:

Input and Initialization:
Gathers user input for the LP problem.
Checks for negative second members in constraints and adjusts them if needed.
Determines the type of problem (MAX or MIN).
Calculates the number of decision variables, surplus variables, and artificial variables.
Generates the initial simplex matrix.
Main Loop:
Finds the entering variable (critical path) using the appropriate objective function.
Finds the leaving variable (pivot row) using the ratio test.
Updates the simplex matrix using pivot operations.
Checks for termination conditions (optimal solution or infeasibility).
Output:
Prints intermediate steps of the Simplex method, including the simplex matrix at each iteration.
Provides the final solution or indicates infeasibility.
Specific Functions:

format_equation: Formats linear equations for printing.
format_matrix: Formats the simplex matrix for printing, potentially using the tabulate module.
print_PL: Prints the initial LP configuration.
print_matrix: Prints the current simplex matrix.
find: Finds the maximum or minimum value in a vector, considering optional filtering for positive or negative values.
find_cp: Finds the critical index in a vector, considering maximum, minimum, and degenerate cases.
clone_matrix: Creates a copy of a matrix.
elemenate_fraction: Rounds numbers to two decimals or to an integer.
Additional Observations:

It uses a variable big_M to track whether the Big M method is needed for handling certain constraint types.
It handles degenerate cases by tracking a degenere flag.
It prompts for user input to pause between iterations, making it interactive for demonstration purposes.
