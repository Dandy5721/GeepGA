import argparse
def parse_args():
    parser=argparse.ArgumentParser(description="None")

    help = "The addresses to connect."
    #add_argument 会自己加上命令行中的值
    #给出参数名字的首先按参数名字匹配，没有的按添加顺序匹配
    #nargs = ? 只用一个  * 0或多 + 至少一个
    parser.add_argument('addresses', nargs='*', help=help)
    #前面没有-表示必须
    help = "The addresses to connect."
    parser.add_argument('filename', help=help)

    help = "The addresses to connect."
    parser.add_argument('-p', '--port', type=int, help=help)

    help = "The addresses to connect."
    parser.add_argument('--iface', help=help, default='localhost')

    help = ''

    parser.add_argument('--delay', type=float, help=help, default=.7)

    help =''
    parser.add_argument('--bytes', type=int, help=help, default=10)

    args = parser.parse_args();
    return args

if __name__:
    args=parse_args()
    print(args)