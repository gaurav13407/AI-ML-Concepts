## MUltiproceesing with proceespoolexceuter

from concurrent.futures import ProcessPoolExecutor
import time

def square_numbers(number):
    time.sleep(1)
    return f"square:{number*number}"

number=[1,2,3,4,5]
if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3)as executor:
        results=executor.map(square_numbers,number)
        
    for result in results:
        print(result)