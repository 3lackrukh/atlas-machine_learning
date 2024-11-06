#!/usr/bin/env python3
from io import StringIO
densenet121 = __import__('7-densenet121').densenet121

if __name__ == '__main__':
    summary_buffer = StringIO()

    model = densenet121(32, 0.5)
    model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
    
    model_summary = summary_buffer.getvalue()
    
    with open('densenet121.txt', 'w') as f:
        f.write(model_summary)