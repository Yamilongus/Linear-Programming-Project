#Team Members:
#Luis Yamil Pagan Tapia
#Angel Martinez
#Course: SICI4028
#Insitution: Universidad de Puerto Rico en Bayamon

# Indicates whether to use the tabulate module for table formatting
USE_TABULATE_MODULE = True 
# Indicates whether to use UTF-8 encoding for superscripts
USE_UTF8_ENCODE = True
# Placeholder for empty cells in tables
EMPTY_CELL = "*"


# Function to convert numbers to superscripts
def get_super(x):
    return str(x)

# If UTF-8 encoding is used, attempt to encode superscripts in UTF-8
if USE_UTF8_ENCODE :
    try:
        from sys import stdout
        '₀₁₂₃₄₅₆₇₈₉'.encode(stdout.encoding)
        def get_super(x):
            normal = "0123456789"
            super_s = "₀₁₂₃₄₅₆₇₈₉"
            res = str(x).maketrans(''.join(normal), ''.join(super_s))
            return str(x).translate(res)

    except UnicodeEncodeError:
        pass
# Find the maximum or minimum value in a vector, optionally filtering for positive or negative values
def find(vect, maxi=True, positif=True):
    new_vect = [i for i in vect if i != None and i < 0]
    if positif:
        new_vect = [i for i in vect if i != None and i > 0]
    if new_vect == []:
        return None
    if maxi:
        return vect.index(max(new_vect))
    return vect.index(min(new_vect))

# Find the critical index in a vector, considering maximum, minimum, and degenerate cases
def find_cp(vect):
    if type == "MAX":
        return find(vect, maxi=True, positif=True)
    if degenere:
        return find(vect, maxi=True, positif=False)
    return find(vect, maxi=False, positif=False)
# Clone a matrix
def clone_matrix(matrix):
    return [[i for i in vect] for vect in matrix]

# Eliminate fractions by rounding numbers to two decimals or to an integer
def elemenate_fraction(nbr):
    try :
        return round(nbr) if round(nbr) == nbr else round(nbr, 2)
    except :
        return nbr

# Format a linear equation from a vector and an operator
def format_equation(vect, op):
    equa = ""
    is_first = True
    for i in range(len(vect)):
        if vect[i] != 0:
            sign = " - " if vect[i] < 0 else " + "
            coeff = "" if vect[i] in [-1, 1] else str(abs(elemenate_fraction(vect[i]))) + "*"
            sign = (sign.strip() if sign == " - " else "") if is_first else sign
            equa += "{}{}x{}".format(sign, coeff, get_super(i + 1))
            is_first = False
    
    return ('0' if equa == '' else equa) + (" ≥" if op > 0 else " ≤" if op < 0 else " =")

# Print the linear program (LP) configuration
def print_PL():
    print("\n"+"-"*40)
    print("{}   {} z".format(type, format_equation(obj_func_coeff, 0)))
    for i in range(m):
        print("{}   {} {}".format("s.c" if i == 0 else "   ",
        format_equation(constraints_coeff[i], constraints_opr[i]),
        elemenate_fraction(constraints_sm[i])))
    print("      ", end="")
    for i in range(n):
        print("x{}".format(get_super(i+1)), end=" ≥ 0\n" if i == n - 1 else ", ")
    print("-"*40, "\n")

# Format the simplex matrix for printing
def format_matrix():
    ADD_SM_CP_COLOMN = True
    COL = col
    new_mat = clone_matrix(simplex_matrix)
    
    # Adjust Big M objective function values if necessary
    if big_M :
        for i in range(COL) :
            nbr = elemenate_fraction(new_mat[-1][i])
            if nbr == None :
                continue
            Mnbr = elemenate_fraction(simplex_Mobj_func[i])
            sign = '' if Mnbr > 0 and nbr == 0 else '-' if Mnbr < 0 else '+'
            Mnbr = '' if Mnbr == 0 else sign + (str(abs(Mnbr)) if Mnbr not in [1, -1] else '') + 'M'
            new_mat[-1][i] = (str(nbr) if nbr != 0 else '' if Mnbr != '' else '0') + Mnbr

                # Add SM/CP column if specified
    if ADD_SM_CP_COLOMN :
        for i in range(row - 1) :
            new_mat[i] += [elemenate_fraction(constraints_sm_cp[i])]
        new_mat[-1].append(EMPTY_CELL)
        COL += 1

    # Convert all elements to string, replacing None with EMPTY_CELL
    for i in range(row):
        for j in range(COL):
            new_mat[i][j] = str(elemenate_fraction(new_mat[i][j])) if new_mat[i][j] != None else EMPTY_CELL

    # Create header row for the matrix
    header = ["x" + get_super(i+1) for i in range(n)] + ["e" + get_super(i+1) for i in range(ecart_nbr)]
    header += ["t" + get_super(i+1) for i in range(art_nbr)]
    header += ["SM", "SM/CP"] if ADD_SM_CP_COLOMN else ["SM"]
    new_mat.insert(0, header)

    # Add base variable (VB) identifiers to the matrix
    for i in range(row - 1) :
        new_mat[i + 1].insert(0, header[vbs[i]])
    new_mat[0].insert(0, "Base")
    new_mat[-1].insert(0, "-z")
    return new_mat

# Print the simplex matrix in a formatted manner
def print_matrix():
    COL = col + 2
    ROW = row + 1
    matrix = format_matrix()
    col_max_len = [0]*(COL)

    for i in range(COL) :
        col_max_len[i] = max(len(row[i]) for row in matrix)

    print()
    for i in range(ROW) :
        print("|", end='')
        for j in range(COL) :
            elem = matrix[i][j]
            print('', elem, ' '*(col_max_len[j] - len(elem)) + '|', end='')
        print()

# Use the tabulate module for printing the matrix, if available
if USE_TABULATE_MODULE :
    try :
        from tabulate import tabulate
        def print_matrix():
            new_mat = format_matrix()
            print(tabulate(new_mat[1:], tablefmt="pretty", headers=new_mat[0]))
        
    except ModuleNotFoundError:
        print("\n\n+ (!) The 'tabulate' module is not installed")
        print("+ (!) This helps draw tables correctly in the terminal.")
        print("+ (!) Exit the program and type 'pip3 install tabulate' on the terminal to install it.")
        input("+ (!) Click ENTER to continue with the predefined table format ")

# User input for the coefficients of the objective function
print("\n+ Choose the type of PL (MAX/MIN)")
inp_err = False
while True :
    type = input("|- "+ ("(!) " if inp_err else '') + "Type the PL : ").upper()
    try : 
        if type not in ["MAX", "MIN"] :
            raise Exception()
        break
    except :
        inp_err = True

print("\n+ Enter the coefficients of the objective function separated by a space")
inp_err = False
while True :
    inp = input("|- "+ ("(!) " if inp_err else '') + "Objective Function : ").split()
    try : 
        obj_func_coeff = [float(i) for i in inp]
        break
    except :
        inp_err = True
# User input for constraint coefficients
constraints_coeff = []
m = 0 # Counter for the number of constraints
print("\n+ Enter the stress coefficients separated by a space (leave empty to exit)")
while True :
    inp_err = False
    while True :
        print("|- "+ ("(!) " if inp_err else '') + "Constraint", m + 1, ": ", end="")
        try : 
            inp = [float(i) for i in input().split()]
            break
        except :
            inp_err = True
    if inp == [] :
        break
    constraints_coeff += [inp]
    m += 1

# User input for constraint operators
constraints_sm = []
print("\n+ Enter the Second Member of Constraints")
for i in range(m) :
    inp_err = False
    while True : 
        print("|- "+ ("(!) " if inp_err else '') + "Constraint", i + 1, ": ", end="")
        try :
            constraints_sm += [float(input())]
            break
        except :
            inp_err = True

constraints_opr = []
print("\n+ Enter the constraint operators (1 for '≤' | 2 for '=' | 3 for '≥')")
for i in range(m) :
    inp_err = False
    while True :
        print("|- "+ ("(!) " if inp_err else '') + "Constraint", i + 1, ": ", end="")
        try :
            inp = int(input())
            if inp not in [1,2,3] :
                raise Exception()
            constraints_opr += [inp - 2]
            break
        except :
            inp_err = True


# Determine the number of decision variables
n = max([len(vect) for vect in constraints_coeff + [obj_func_coeff]]) 

# Adjust the length of constraints and objective function coefficients
constraints_coeff = [constraint + [0]*(n - len(constraint)) for constraint in constraints_coeff]
obj_func_coeff += [0]*(n - len(obj_func_coeff))

# Print the initial LP configuration
print_PL()

# Check if constraints need to be adjusted (if any second member is negative)
is_constraints_changed = False
for i in range(m):
    if constraints_sm[i] < 0:
        is_constraints_changed = True
        constraints_sm[i] *= -1
        constraints_coeff[i] = [-i for i in constraints_coeff[i]]
        constraints_opr[i] *= -1

# If constraints were changed, print the modified LP configuration
if is_constraints_changed :
    print("(!) The second members must be positive")
    print_PL()

# Check if the LP can be solved with the Simplex method
big_M = any([False if opr == -1 else True for opr in constraints_opr])

# Calculate the number of artificial and surplus variables to be added
art_nbr = 0 #NUmber of artificial variables
ecart_nbr = m # NUmber of surplus variables
for opr in constraints_opr :
    art_nbr += 1 if opr != -1 else 0
    ecart_nbr -= 1 if opr == 0 else 0

# Notify the method to be used for solving the LP
if big_M:
    print("(!) The second members must be positive\n")
else :
    print("(!) We will solve this PL with the Simplex method\n")
input("+ Click ENTER to resolve this PL: ")

# Generate the surplus and artificial variables matrices
ecart_coeff = [[0 for i in range(ecart_nbr)] for i in range(m)]
art_coeff = [[0 for i in range(art_nbr)] for i in range(m)]

tmp_e, tmp_a = 0, 0
for i in range(m) :
    if constraints_opr[i] != 0 :
        ecart_coeff[i][tmp_e] = - constraints_opr[i]
        tmp_e += 1
    if constraints_opr[i] != -1:
        art_coeff[i][tmp_a] = 1
        tmp_a += 1

obj_func_ecoeff = [0]*ecart_nbr
obj_func_acoeff = [0]*art_nbr

# Calculate the new Big M objective function
Mobj_func_coeff  = [0]*n
Mobj_func_ecoeff = [0]*ecart_nbr
Mobj_func_acoeff  = [0]*art_nbr
Mobj_func_sm = 0

for i in range(m) :
    if constraints_opr[i] == -1:
        continue
    for j in range(n):
        Mobj_func_coeff [j] += -constraints_coeff[i][j]
    
    for j in range(ecart_nbr):
        Mobj_func_ecoeff[j] += -ecart_coeff[i][j]
    
    Mobj_func_sm += constraints_sm[i]

# Adjust Big M objective function for MAX problems
if type == "MAX":
    Mobj_func_coeff  = [-i for i in Mobj_func_coeff ]
    Mobj_func_ecoeff = [-i for i in Mobj_func_ecoeff]
    Mobj_func_sm *= -1

Mobj_func_sm *= -1

# Determine the initial basic variables (VBs)
vbs = [None] * m
tmp_e, tmp_t = 0, 0
for i in range(m):
    if constraints_opr[i] == -1:
        vbs[i] = n + tmp_e
        tmp_e += 1
        continue
    else :
        vbs[i] = n + ecart_nbr + tmp_t
        tmp_t += 1

# Generate the initial simplex matrix
simplex_matrix = clone_matrix(constraints_coeff)
for i in range(m) :
    simplex_matrix[i] += ecart_coeff[i] + art_coeff[i] + [constraints_sm[i]]
simplex_matrix += [[i for i in obj_func_coeff]]
simplex_matrix[-1] += obj_func_ecoeff + obj_func_acoeff +  [0]

# Generate the Big M objective function for the simplex matrix
simplex_Mobj_func = Mobj_func_coeff  + Mobj_func_ecoeff + Mobj_func_acoeff  + [Mobj_func_sm]

# Initialize matrix dimensions and a flag for degenerate cases
col = n + ecart_nbr + art_nbr + 1
row = m + 1

degenere = False
k = 0

# Initialize arrays for storing the ratio of the Second Member to the Critical Path
constraints_sm_cp  = [0]*m
obj_func_sm_cp = 0

while True :

    if big_M :
        cp = find_cp(simplex_Mobj_func[: n + ecart_nbr])
    if not(big_M) or (big_M and (cp == None and all(not i for i in simplex_Mobj_func[: n + ecart_nbr]))):
        cp = find_cp(simplex_matrix[-1][: n + ecart_nbr])
    
    if cp == None:
        break

    for i in range(m):
        if simplex_matrix[i][cp] == 0 :
            constraints_sm_cp[i] = None
            continue
        constraints_sm_cp[i] = simplex_matrix[i][-1] / simplex_matrix[i][cp]
    
    # Trouver la ligne de pivot
    lp = find(constraints_sm_cp, maxi=False, positif=True)

    ###############
    print("\nK =", k)
    print_matrix()
    ##############

    if lp == None:
        print("(!) The solution is infinite")
        exit(1)
    
    # Trouver le pivot
    pivot = simplex_matrix[lp][cp]

    print("\ncp =", cp + 1, end="  |  ")
    print("lp =", lp + 1, end="  |  ")
    print("pivot =", elemenate_fraction(pivot))
    print("-" * 34, "\n")
    input("+ Click ENTER to continue: ")

    vbs[lp] = cp
    k += 1

    simplex_matrix[lp] = [nbr / pivot if nbr != None else None for nbr in simplex_matrix[lp]]

    old_mat = clone_matrix(simplex_matrix)
    for i in range(row):
        if i == lp:
            continue
        for j in range(col):
            try :
                simplex_matrix[i][j] = old_mat[i][j] - old_mat[i][cp] * old_mat[lp][j]
            except :
                pass

    if big_M:
        old_Mobj_func = simplex_Mobj_func.copy()
        for i in range(col):
            try :
                simplex_Mobj_func[i] = old_Mobj_func[i] - old_Mobj_func[cp] * old_mat[lp][i]
            except :
                pass

        for i in range(n + ecart_nbr, n + ecart_nbr + art_nbr):
            if i not in vbs:
                for j in range(row):
                    simplex_matrix[j][i] = None
                simplex_Mobj_func[i] = None
    
    for i in range(row - 1):
        if (simplex_matrix[i][-1] == 0):
            print("\n(!) The solution is degenerate")
            print_matrix()

            if degenere:
                print("(!) We cannot apply Bland's rule for the second time...\n")
                exit(1)

            input("\n+ Click ENTER to apply Bland's rule: ")
            degenere = True
            simplex_matrix = clone_matrix(old_mat)
            if big_M:
                simplex_Mobj_func = old_Mobj_func.copy()
            break

print("\nK =", k)
print_matrix()
print("\nAll the coefficients of the objective function are {} then the stopping criterion is verified\n".format("negative" if type == "MAX" else "positive"))

optimal_sol = [0] * n
for i in range(m):
    if vbs[i] < n:
        optimal_sol[vbs[i]] = elemenate_fraction(simplex_matrix[i][-1])

print("The optimal solution :\nx* = (", end="")
for i in range(n):
    print(optimal_sol[i], end="" if i == n - 1 else ", ")
print(")\n")

print("The best :\nz* =", -elemenate_fraction(simplex_matrix[-1][-1]), "\n")