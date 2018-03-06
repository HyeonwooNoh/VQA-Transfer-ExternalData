def clean_description(string):
    string = string.lower()
    string = string.replace('\n', '')
    string = string.replace(' re', ' are')
    string = string.replace('.', ' .')  # '  ' to ' ' should follow this.
    string = string.replace('   ', ' ')
    string = string.replace('  ', ' ')
    if string == '':
        return string
    if string == ' ':
        string = ''
        return string
    if string[-1] == ' ':
        string = string[:-1]
    if string[0] == ' ':
        string = string[1:]
    return string
