import random, string, sys
print(sys.version)

def random_str(num, length = 7):
    chars = string.ascii_letters + string.digits
    with open("Running_result.txt", 'w') as f:
        for i in range(num):        
            s = [random.choice(chars) for i in range(length)]
            f.write(''.join(s) + '\n')

if __name__ == '__main__':
    random_str(200)