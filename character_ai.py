from characterai import aiocai
import asyncio
from telegram_mes import send_telegram_message

class CharacterAIClient:
    def __init__(self, api_key, char_id):
        self.api_key = api_key
        self.char_id = char_id
        self.client = aiocai.Client(api_key)
        self.me = None

    async def initialize(self):
        self.me = await self.client.get_me()
        self.chat = await self.client.connect()
        self.new, self.answer = await self.chat.new_chat(self.char_id, self.me.id)

    async def get_action(self, caption):
        message = await self.chat.send_message(self.char_id, self.new.chat_id, caption)
        out_text = message.text
        send_telegram_message(out_text)
        return out_text

    async def close(self):
        await self.chat.close()

async def async_get_action(client, caption):
    return await client.get_action(caption)

def weapon_detected_c(weapon):
    caption = f'there is a {weapon}. what should i do? give me only 1 short sentence.'
    
    client = CharacterAIClient('3206be540ccc00da1242687f6e42e81d8335464f', '8edXgZQQSqE5BowLEr72JbasHBC2Yd6LvnmPMEmemfA')

    async def run():
        await client.initialize()
        response = await async_get_action(client, caption)
        await client.close()
        return response

    return asyncio.run(run())

"""
def weapon_detected_c(weapon):
    caption = f'there is a {weapon}. what should i do? give me only 1 short sentence.'
    async def async_get_action(caption):
        char = '8edXgZQQSqE5BowLEr72JbasHBC2Yd6LvnmPMEmemfA'
        client = aiocai.Client('3206be540ccc00da1242687f6e42e81d8335464f')

        try:
            me = await client.get_me()
        except Exception as e:
            print(f"Error getting user information: {e}")
            return

        async with await client.connect() as chat:
            new, answer = await chat.new_chat(char, me.id)
            message = await chat.send_message(char, new.chat_id, caption)
            out_text = message.text
            print(f'{message.name}: {out_text}')
            send_telegram_message(out_text) 
            return message.text

    return asyncio.run(async_get_action(caption))

def weapon_detected_c(weapon):
    out_text = get_action_from_characterai(f'there is a {weapon}. what should i do? give me only 1 short sentence')
    #print(f">{out_text}\n\n")
    send_telegram_message(out_text) 

"""