'''
    Grading script for GPU Project 4 - Flash Attention

    This script assumes that the virtual environment is setup and 
    activated. (i.e. you can run:
    
        - python -m task2.compile and 
        - python -m task2.test
    
    and they work.
'''
import sys
import subprocess

output_file="results.txt"

def venv_is_inactive():
    # Check for venv status
    if sys.prefix == sys.base_prefix:        
        print("***** Please activate the virtual environment and try again *****")
        return True
    return False

def grade_task1():
    print("Grading Task 1: ", end="")

    t1_score=2
    res = subprocess.run( ["python", "-m", "task1.test"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if "All tests passed" in res.stdout:
        print(f"Passed: {t1_score} points")
        return t1_score
    
    print("Failed")
    return 0   
   
def grade_task2():
    print("Compiling Task 2: ", end="")

    t2_score=3
    res = subprocess.run( ["python", "-m", "task2.compile"], capture_output=True, text=True)

    if "Failed to build" in res.stderr:
        #Compilation error
        print("Failed")
        return 0
    else:
        print("Passed")

    #Compilation was successful so run task2.test
    print("Grading Task 2: ", end="")
    res = subprocess.run( ["python", "-m", "task2.test"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if "All tests passed" in res.stdout:
        print(f"Passed: {t2_score} points")
        return t2_score
    
    print("Failed")
    return 0

def grade_task3():
    print("Grading Task 3: ", end="")

    t3_score=3
    res = subprocess.run( ["python", "-m", "task3.test"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if "All tests passed" in res.stdout:
        print(f"Passed: {t3_score} points")
        return t3_score
    
    print("Failed")
    return 0
   
def grade_task4():
    print("Compiling Task 4: ", end="")

    t4_score=4
    res = subprocess.run( ["python", "-m", "task4.compile"], capture_output=True, text=True)

    if "Failed to build" in res.stderr:
        #Compilation error
        print("Failed")
        return 0
    else:
        print("Passed")

    #Compilation was successful so run task4.test
    print("Grading Task 4: ", end="")
    res = subprocess.run( ["python", "-m", "task4.test"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if "All tests passed" in res.stdout:
        print(f"Passed: {t4_score} points")
        return t4_score
    
    print("Failed")
    return 0

def grade_task5():
    print("Compiling Task 5: ", end="")

    t5_score=4
    res = subprocess.run( ["python", "-m", "task5.compile"], capture_output=True, text=True)

    if "Failed to build" in res.stderr:
        #Compilation error
        print("Failed")
        return 0
    else:
        print("Passed")

    #Compilation was successful so run task4.test
    print("Grading Task 5: ", end="")
    res = subprocess.run( ["python", "-m", "task5.test"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if "All tests passed" in res.stdout:
        print(f"Passed: {t5_score} points")
        return t5_score
    
    print("Failed")
    return 0

def grade_task6():
    print("Grading Task 6: ", end="")

    t6_score=3
    res = subprocess.run( ["python", "-m", "task6.test"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if "Prefill time" in res.stdout and "decode time" in res.stdout and "after decode: 1124" in res.stdout:
        print(f"Passed: {t6_score} points")
        return t6_score
    
    print("Failed")
    return 0

def grade_task7():
    # Grading done only for functionality (did it generate output) and not for
    # full coherency or useful information.
    print("Grading Task 7: ", end="")

    t7_score=1
    res = subprocess.run( ["python", "-m", "task7.inference"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if "flashattention" in res.stdout and "kernels" in res.stdout:
        print(f"Passed: {t7_score} points")
        return t7_score
    
    print("Failed")
    return 0

    # print("--------------------- STDOUT ---------------------")
    # print(res.stdout)
    # print("--------------------- STDERR ---------------------")
    # print(res.stderr)
    # print("Testing Task 2")
    # return 

def main():
    print("Grading Project")
    total_score=0
    task4_score=task5_score=0

    # Grade Tasks
    total_score += grade_task1()
    total_score += grade_task2()
    total_score += grade_task3()

    task4_score = grade_task4()
    total_score += task4_score

    task5_score = grade_task5()
    total_score += task5_score

    total_score += grade_task6()
    total_score += grade_task7()
    
    # No longer true - see the grade_task7() function
    # for details. Task 7 is a free point if tasks 4 and 5 are successful
    # if (task5_score + task5_score) == 8:
    #     print("Grading Task 7: Passed")
    #     total_score += 1    # Task 7 is worth one point
    # else:
    #     print("Grading Task 7: Failed")

        
    return total_score                            


if __name__ == "__main__":
    # Exit if venv is not activated
    if venv_is_inactive(): sys.exit()
        
    score=main()
    print(f"Total score: {score}")