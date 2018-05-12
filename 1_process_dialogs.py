# coding=utf-8
# -*- coding: utf-8 -*-

import pickle
import re
import sys

from util_preprocessing import *

if not os.path.exists(config.dialogs_dir):
    sys.exit(config.dialogs_dir + " dir is not exists")

dialog_files_names = os.listdir(config.dialogs_dir)

if len(dialog_files_names) == 0:
    sys.exit("files not found in " + config.dialogs_dir)

user_name = u"Вячеслав Корниенко"
dialog_users = map(lambda d: d[:d.index(".txt")].decode("utf-8"),
                   [name for name in dialog_files_names if "txt" in name])


class Message:
    owner = ""
    text = ""

    def __init__(self, owner, text):
        self.owner = owner
        self.text = text
        pass


def parse_messages(file_name):
    dialog_file = codecs.open(config.dialogs_dir + "/" + file_name, encoding="utf8")
    text = dialog_file.read()
    print "parse: ", file_name

    # regex = re.compile("MESSAGE_START>(\S+ \S+) \((\d+:\d+:\d+  \d+/\d+/\d+)\):([\s\S]+?)<MESSAGE_END")
    # messages = map(lambda x: Message(x.group(1), x.group(3)[2:]), regex.finditer(text))

    regex = re.compile("(.+?)%%(.+)")
    messages = []
    for match in regex.finditer(text):
        messages.append(Message("Person one", match.group(1)))
        messages.append(Message("Person two", match.group(2)))
    return messages


def clean_message(message):
    text = message.text
    text = text.replace(u"ё", u"е")
    text = re.sub(u"[^А-я ]+", u" ", text)
    text = re.sub(u" {2,}", u" ", text)
    text = text.lower()
    text = text.strip()
    message.text = text
    return message


def collapse_messages(msgs):
    indexes_to_remove = []

    def process(message, index):
        if index + 1 <= len(msgs) - 1:
            next_message = msgs[index + 1]
            if message.owner == next_message.owner:
                message.text = message.text + " " + next_message.text
                indexes_to_remove.append(index + 1)
        return message

    while True:
        indexes_to_remove[:] = []
        msgs = map(lambda (i, m): process(m, i), enumerate(msgs))

        for index in reversed(indexes_to_remove):
            msgs.pop(index)

        if not indexes_to_remove:
            break

    return msgs


def filter_messages(msgs):
    return [m for m in msgs if (m.owner in dialog_users or m.owner == user_name)]


def messages_to_question_answer(msgs):
    out = list()
    if len(msgs) > 0:
        my = u""
        other = u""
        if msgs[0].owner == user_name:
            msgs.pop(0)
        for i, msg in enumerate(msgs):
            if i % 2 == 0:
                my = msg.text
            else:
                other = msg.text
            if my and other:
                out.append([
                    add_service_tokens(tokenize(my), config.max_words),
                    add_service_tokens(tokenize(other), config.max_words, end=True)
                ])
                my = None
                other = None
    return out


def save_messages(data):
    with codecs.open(config.processed_messages_file, 'w+b', encoding='utf-8') as f:
        pickle.dump(data, f)


all_messages = []
dialog_messages = []

for dialog_name in dialog_files_names:
    messages = parse_messages(dialog_name)
    # messages = filter_messages(messages)
    messages = [x for x in map(lambda y: clean_message(y), messages) if len(x.text) != 0]
    messages = [x for x in messages if len(tokenize(x.text)) <= 30]
    messages = collapse_messages(messages)
    messages = messages_to_question_answer(messages)
    all_messages += messages

print u'сохраняем результат', str(len(all_messages))
save_messages(all_messages)
print(u'готово')
