def calculator(num : str) -> str:
    """This tool is used to perform finanical calculations. The input is a string that represents a mathematical expression (111 * 5.54), and the output is the result of the calculation."""

    check = set("0123456789+-*/.()%, ")

    if not all(c in check for c in num):
        return "Invalid input. Only numbers and basic mathematical operators are allowed."  
    
    try:
        result = eval(num, {"__builtins__": {}})
        return str(round(result, 4))
    
    except:
        return "Error in calculation. Please check the expression and try again."