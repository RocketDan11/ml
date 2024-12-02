import yfinance as yf
import anthropic
import mouse
from PIL import ImageGrab

def get_stock_price(stock_symbol):
    try:
        stock = yf.Ticker(stock_symbol)
        return stock.fast_info['lastPrice']
    except Exception as e:
        return f"Error fetching price for {stock_symbol}: {str(e)}"

print(get_stock_price('AAPL'))

def gen_click_event(x, y):
    mouse.click(x, y)   

def gen_screenshot():
    return ImageGrab.grab()


tool_definition = {
    "name": "get_stock_price",
    "description": "Retrieves the current stock price for a given company",
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {
                "type": "string",
                "description": "The company name to fetch stock data for"
            }
        },
        "required": ["company"]
    }
}

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "what is the stock price of apple right now ?"}],
    max_tokens=500,
    tools=[tool_definition]
)

# Handle the tool calls and execute them
if response.content[0].type == 'tool_calls':
    tool_calls = response.content[0].tool_calls
    for tool_call in tool_calls:
        if tool_call.function.name == 'get_stock_price':
            # Get the stock price using the function
            stock_price = get_stock_price(tool_call.function.arguments['company'])
            
            # Send the result back to Claude
            final_response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {"role": "user", "content": "what is the stock price of apple right now ?"},
                    {"role": "assistant", "content": None, "tool_calls": response.content[0].tool_calls},
                    {"role": "tool", "content": str(stock_price), "tool_call_id": tool_call.id}
                ],
                max_tokens=500,
            )
            
            print(final_response.content[0].text)  # This will print Claude's final response with the stock price