import os
import logging
import requests
import asyncio
import websockets
import json
import base58
import base64
import re
import time
#from flask import Flask, request, jsonify
from quart import Quart, request, jsonify
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders import message
from asyncstdlib import enumerate
from solana.rpc.websocket_api import connect
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
from solana.rpc.commitment import Processed
# from solana.transaction import Transaction
from solders.signature import Signature  # Import Signature
#from solana.keypair import Keypair
from dotenv import load_dotenv
import ast  # This will be used to safely convert the string to a list
from solders.instruction import Instruction

from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import transfer_checked, TransferCheckedParams

from solana.rpc.commitment import Confirmed
from solana.rpc.api import Client
from solana.rpc.types import TxOpts
# from solana.keypair import Keypair
# from solana.publickey import PublicKey
from solders.pubkey import Pubkey
# from solana.transaction import Transaction
import solana
from solders.instruction import AccountMeta, Instruction as TransactionInstruction
# from solders.utils import b58decode
from base58 import b58decode
from crypto_tracker import CryptoPLTracker
import subprocess
# Import the restart_server function from the restart_module.py file
from my_bot_environment import restart_server
from market_trend_analysis import identify_trend, get_latest_trend
from transaction_tracker import parse_transaction_log, calculate_pl, calculate_pl_last_n_hours, sum_pl_last_n_hours
from solana.rpc.async_api import AsyncClient
from quart_cors import cors

# Replace with your Chrome extension's origin
CHROME_EXTENSION_ORIGIN = "chrome-extension://beamfeaeoojdfafafeclgeijihdcbaed"

app = Quart(__name__)
# Restrict CORS to only allow requests from your Chrome extension
app = cors(app, allow_origin=[CHROME_EXTENSION_ORIGIN])


# Load environment variables
load_dotenv()

#app = Flask(__name__)
# app = Quart(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create loggers for different modules or files
file_logger = logging.getLogger("file_logger")
console_logger = logging.getLogger("console_logger")
transaction_logger = logging.getLogger("transaction_logger")  # New logger for transaction_log

# Set log levels (optional)
file_logger.setLevel(logging.INFO)  # Log to file only INFO and above
console_logger.setLevel(logging.DEBUG)  # Log to console DEBUG and above
transaction_logger.setLevel(logging.INFO)  # Log for transaction_log at INFO level

# Create file handler for logging to the original file (file_log.log)
file_handler = logging.FileHandler("file_log.log")
file_handler.setLevel(logging.INFO)

# Create file handler for logging to the new file (transaction_log.log)
transaction_handler = logging.FileHandler("transaction_log.log")
transaction_handler.setLevel(logging.INFO)  # You can adjust the level if necessary

# Create console handler for logging to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create log formatters
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
transaction_formatter = logging.Formatter("%(asctime)s - %(levelname)s - Transaction: %(message)s")  # Custom format for transaction_log
console_formatter = logging.Formatter("%(levelname)s - %(message)s")

# Add formatters to handlers
file_handler.setFormatter(file_formatter)
transaction_handler.setFormatter(transaction_formatter)  # Apply transaction format
console_handler.setFormatter(console_formatter)

# Add handlers to loggers
file_logger.addHandler(file_handler)
transaction_logger.addHandler(transaction_handler)  # Add handler for transaction_log
console_logger.addHandler(console_handler)

# Initialize a global lock
action_lock = asyncio.Lock()

# Flags to track the state of buy and sell actions
buy_waiting = False
sell_waiting = False
buy_active = False
sell_active = False
buy_task = None
sell_task = None

# Environment variables
JUPITER_API_KEY = os.getenv('JUPITER_API_KEY')
WALLET_ID = os.getenv('WALLET_ID')
PRIVATE_KEY = os.getenv('PRIVATE_KEY')
# Retrieve environment variables
SWAP_AMOUNT = os.getenv('SWAP_AMOUNT')
RETRY_COUNT = os.getenv('RETRY_COUNT')

# Convert to appropriate types with error handling
try:
    SWAP_AMOUNT = float(SWAP_AMOUNT) if SWAP_AMOUNT is not None else 0.1
except ValueError:
    SWAP_AMOUNT = 0.1  # Default value if conversion fails
    print("Invalid SWAP_AMOUNT value. Using default: 0.1")

try:
    RETRY_COUNT = int(RETRY_COUNT) if RETRY_COUNT is not None else 1
except ValueError:
    RETRY_COUNT = 1  # Default value if conversion fails
    print("Invalid RETRY_COUNT value. Using default: 1")

SOL_TICKER = "So11111111111111111111111111111111111111112"  # SOL token ID
TOKEN_DECIMALS = 10**9  # For Solana (9 decimals)
DESIRED_CRYPTO_TICKER = None
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL")  # Solana RPC URL
# CoinMarketCap API Key
CMC_API_KEY = "your_api_key_here"

# Build the transaction
payer = Pubkey.from_string(WALLET_ID)
owner = payer

# Constants
TOKEN_PROGRAM_ID = Pubkey(base58.b58decode("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"))

SYS_PROGRAM_ID = Pubkey(base58.b58decode("11111111111111111111111111111111"))
SYSVAR_RENT_PUBKEY = Pubkey(base58.b58decode("SysvarRent111111111111111111111111111111111"))
associated_token_address = Pubkey.from_string("D2z2ZVyKRnG5e2CCHvtoTRW6oKVyA3KnQZWrCLYjqwoA")

# TOKEN_PROGRAM_ID = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
# ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPv8Zhb32s98e7EfH9SSz4p2BLxDwpY9GQK2e8d"
# SYS_PROGRAM_ID = Pubkey.from_string("11111111111111111111111111111111")
# SYSVAR_RENT_PUBKEY = Pubkey.from_string("SysvarRent111111111111111111111111111111111")

### Decode the base58 string to bytes
#base58_string = "ComputeBudget111111111111111111111111111111111111"
#decoded_bytes = base58.b58decode(base58_string)
#
## Create the Pubkey from the decoded bytes
#COMPUTE_BUDGET_PROGRAM_ID = Pubkey.from_bytes(decoded_bytes)

# Initialize Solana client and Keypair
#client = Client(SOLANA_RPC_URL)
# client = Client("https://empty-delicate-aura.solana-mainnet.quiknode.pro/eda79e548ae61382c63ac4e795fdd096ef2f24dd")
client = Client("https://api.mainnet-beta.solana.com")

# private_key_list = ast.literal_eval(PRIVATE_KEY)
# # Step 2: Convert to a bytes object
# private_key_bytes = bytes(private_key_list)
# # Base58 encode the byte array
# encoded_private_key = base58.b58encode(private_key_bytes).decode('utf-8')
#
# PRIVATE_KEY_PAIR = Keypair.from_bytes(base58.b58decode(encoded_private_key))


PRIVATE_KEY_PAIR = Keypair.from_bytes(base58.b58decode(PRIVATE_KEY))



# Function to get the crypto count (not lamports)
def get_crypto_count(swap_quote, token_decimals):
    # Extract amounts and mint addresses
    input_mint = swap_quote.get("inputMint")
    output_mint = swap_quote.get("outputMint")
    in_amount = int(swap_quote.get("inAmount", 0))
    out_amount = int(swap_quote.get("outAmount", 0))

    # Default result dictionary
    result = {
        #        "sol_sent": None,
        "sol": None,
        "sol_lamports": None,
        #        "non_sol_sent": None,
        "non_sol": None,
        "non_sol_lamports": None

    }

    # Handle SOL
    if input_mint == SOL_TICKER:
        result["sol"] = in_amount / 10**9  # Convert lamports to SOL
        result["sol_lamports"] = in_amount
    elif output_mint == SOL_TICKER:
        result["sol"] = out_amount / 10**9  # Convert lamports to SOL
        result["sol_lamports"] = out_amount

    # Handle Non-Solana Token (sent or received)
    if input_mint != SOL_TICKER:
        result["non_sol"] = in_amount / 10**token_decimals  # Convert to crypto count (not lamports)
        result["non_sol_lamports"] = in_amount
    if output_mint != SOL_TICKER:
        result["non_sol"] = out_amount / 10**token_decimals  # Convert to crypto count (not lamports)
        result["non_sol_lamports"] = out_amount

    return result

def get_sol_amount(swap_quote):
    # Extract amounts
    input_mint = swap_quote.get("inputMint")
    output_mint = swap_quote.get("outputMint")
    in_amount = int(swap_quote.get("inAmount", 0))
    out_amount = int(swap_quote.get("outAmount", 0))

    # Determine if SOL is involved and return the amount in SOL (float)
    if input_mint == SOL_TICKER:
        return in_amount / 10**9  # Convert lamports to SOL
    elif output_mint == SOL_TICKER:
        return out_amount / 10**9  # Convert lamports to SOL
    else:
        return 0  # Return 0 if SOL is not involved

def get_usd_price(symbol):
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": symbol, "vs_currencies": "usd"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data[symbol]["usd"]
    else:
        print(f"Error: {response.text}")
        return None

# Function to get native SOL balance
def get_native_sol_balance(wallet_id):
    logger.info(f"Retrieving native SOL balance for wallet {wallet_id}")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBalance",
        "params": [wallet_id]
    }
    try:
        response = requests.post(SOLANA_RPC_URL, json=payload)
        response_data = response.json()
        sol_balance = response_data.get("result", {}).get("value", 0)
        sol_balance_in_sol = sol_balance / 1e9  # Convert lamports to SOL
        logger.info(f"SOL Balance: {sol_balance_in_sol} SOL")
        return sol_balance_in_sol
    except Exception as e:
        logger.error(f"Error retrieving SOL balance: {str(e)}")
        return 0

# Function to get wallet balance for a specific token
def get_wallet_balance(wallet_id, token_mint_address):
    logger.info(f"Retrieving balance for wallet {wallet_id} and token {token_mint_address}")
    payload = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "getTokenAccountsByOwner",
        "params": [
            wallet_id,
            {"mint": token_mint_address},
            {"encoding": "jsonParsed"},
        ],
    }
    try:
        response = requests.post(SOLANA_RPC_URL, json=payload)
        response_data = response.json()
        logger.info(f"Balance for {token_mint_address} response: {response_data}")
        if response_data["result"]["value"]:
            token_amount = response_data["result"]["value"][0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"]
            logger.info(f"Balance for {token_mint_address}: {token_amount}")
            return token_amount
        else:
            logger.warning(f"No token account found for wallet {wallet_id} and token {token_mint_address}")
            return 0
    except Exception as e:
        logger.error(f"Error retrieving balance: {str(e)}")
        return 0

def get_prioritization_fee_lamports(action="buy"):
    """
        Fetches recent prioritization fees from the Solana RPC endpoint and calculates the prioritizationFeeLamports.

        Args:
        action (str): The action to determine the fee multiplier. Default is "buy". For "sell", a higher multiplier is used.

        Returns:
        int: The calculated prioritizationFeeLamports.
        """
    try:
        # Prepare the payload for the `getRecentPrioritizationFees` method
        payload = {
            "jsonrpc": "2.0",
            "method": "getRecentPrioritizationFees",
            "params": [],
            "id": 1
        }

        # Set headers (optional, depending on your QuickNode setup)
        headers = {
            "Content-Type": "application/json"
        }

        # Send the request to QuickNode
        response_fees = requests.post(SOLANA_RPC_URL, headers=headers, data=json.dumps(payload))

        logger.info(f"Recent Prioritization Fees Response: {response_fees.status_code}")

        # Check the response status
        if response_fees.status_code == 200:
            data = response_fees.json()
            if 'result' in data:
                # Extract the prioritizationFee values
                fees = [entry['prioritizationFee'] for entry in data['result']]
                logger.info(f"Recent Prioritization Fees: {fees}")

                # Calculate the average
                average_fee = sum(fees) / len(fees) if fees else 0
                logger.info(f"Average Fee: {average_fee}")

                # Determine the value of prioritizationFeeLamports
                if average_fee < 500000:
                    prioritizationFeeLamports = 500000
                else:
                    if average_fee > 1000000:
                        average_fee = 1000000
                    prioritizationFeeLamports = int(average_fee)

                logger.info(f"The prioritizationFeeLamports is: {prioritizationFeeLamports}")
                return prioritizationFeeLamports
            else:
                logger.error(f"Error in response: {data}")
        else:
            logger.error(f"Request failed with status code {response_fees.status_code}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

    # Default return if something goes wrong
    return 500000

# Function to fetch decimals for a token
def get_token_decimals(token_mint):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenSupply",
        "params": [token_mint]
    }

    try:
        response = requests.post(SOLANA_RPC_URL, json=payload)
        if response.status_code == 200:
            response_data = response.json()
            decimals = response_data['result']['value']['decimals']
            logger.info(f"Decimals for token {token_mint}: {decimals}")
            return decimals
        else:
            logger.error(f"Failed to fetch decimals for token {token_mint}. HTTP status: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching token decimals for {token_mint}: {str(e)}")
        return None

# Function to request a swap quote from Jupiter API, with up to 5 immediate retries if the call fails
def get_swap_quote_response(input_mint, in_amount, output_mint, slippage_percentage, decimals):
    if in_amount <= 0:
        logger.error("Invalid in_amount: must be greater than 0.")
        return 0

    # Use the dynamically retrieved decimals to calculate lamports
    lamports = int(in_amount * (10 ** decimals))

    # Set a specific slippage tolerance in basis points
    slippage_bps = 100 * slippage_percentage # 1% tolerance
    slippage_bps = int(slippage_bps)
    QUOTE_URL = f"https://quote-api.jup.ag/v6/quote?inputMint={input_mint}&outputMint={output_mint}&amount={lamports}&slippageBps={slippage_bps}"
    logger.info(f"Swap quote: {QUOTE_URL}")
    #    QUOTE_URL = f"https://quote-api.jup.ag/v6/quote?inputMint={input_mint}&outputMint={output_mint}&amount={lamports}"

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = requests.get(url=QUOTE_URL)
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"Swap quote response: {response_data}")
                return response_data
            else:
                logger.error(f"Failed to get quote (status code {response.status_code}), attempt {attempt + 1} of {max_retries}")
        except Exception as e:
            logger.error(f"Error fetching swap quote on attempt {attempt + 1} of {max_retries}: {str(e)}")

    # If all retries fail, return 0 or an error indication
    logger.error("All attempts to fetch swap quote have failed.")
    return 0

# Function to perform swap
# @current_app.task
async def perform_swap(action, slippage_percentage, attempt, percentage=None):
    logger.info(f"Performing swap action: {action}")

    # Get balances
    # sol_balance = get_native_sol_balance(WALLET_ID)
    # crypto_balance = get_wallet_balance(WALLET_ID, DESIRED_CRYPTO_TICKER)

    order_size = "full"

    if action == "buy":
        # sol_balance = get_native_sol_balance(WALLET_ID)
        # # Determine the amount to swap
        # if sol_balance > 0.5:
        #     amount_to_swap = SWAP_AMOUNT  # Use 0.5 SOL if balance is greater than 0.5
        # else:
        #     amount_to_swap = sol_balance * 0.25  # Otherwise, use 50% of the balance
        amount_to_swap = percentage

        order_size = f"{amount_to_swap:.2f} SOL"

        logger.info(f"Amount to swap (buy): {order_size}")
        input_mint = SOL_TICKER
        output_mint = DESIRED_CRYPTO_TICKER
    elif action == "sell":
        crypto_balance = get_wallet_balance(WALLET_ID, DESIRED_CRYPTO_TICKER)

        if crypto_balance == 0:
            logger.error(f"{DESIRED_CRYPTO_TICKER} balance is 0. Cannot perform sell action.")
            return {"error": f"Insufficient {DESIRED_CRYPTO_TICKER} balance to perform sell action."}

        # Determine the amount to swap for selling
        if percentage is not None:
            amount_to_swap = crypto_balance * percentage  # Use specified percentage
            if percentage < 1.00:
                order_size = 'partial'
        else:
            amount_to_swap = crypto_balance * 1.00  # Default to 100%

        logger.info(f"Amount to swap (sell): {amount_to_swap} {DESIRED_CRYPTO_TICKER}")
        input_mint = DESIRED_CRYPTO_TICKER
        output_mint = SOL_TICKER
    else:
        logger.error(f"Invalid action: {action}")
        return {"error": "Invalid action specified"}

    # Get the prioritizationFeeLamports
    # prioritizationFeeLamports = get_prioritization_fee_lamports(action)
    # prioritizationFeeLamports = 1000000

    # Fetch decimals for the input token
    decimals = get_token_decimals(input_mint)

    # Get the swap quote
    quote_response = get_swap_quote_response(input_mint, amount_to_swap, output_mint, slippage_percentage, decimals)

    # Check if quote_response is zero, None, or an empty dictionary
    if not quote_response:  # This checks for falsy values like 0, None, empty dict, etc.
        logger.error("Quote response is zero or invalid. Cannot proceed with the swap.")
        return {"status": "error", "message": "Invalid quote response"}
    else:
        logger.info(f"Quote response received: {quote_response}")

       # # Extract the crypto amounts
       # crypto_count = get_crypto_count(quote_response)
       #
       # # Get Solana's USD price
       # usd_sol_price = get_usd_price("solana")
       #
       # # Calculate USD price of the non-Solana token
       # if crypto_count.get("sol") and crypto_count.get("non_sol"):
       #     usd_non_sol_price = (usd_sol_price * crypto_count["sol"]) / crypto_count["non_sol"]
       #     logger.info(f"USD price of the non-Solana token: {usd_non_sol_price}")
       # else:
       #     logger.info("Error: Missing Solana or non-Solana amounts in the swap quote.")

    #    # Calculate USD value of wallet before transaction
    #    usd_wallet_value_before = (usd_sol_price * sol_balance) + (usd_non_sol_price * crypto_balance)

    action_aligned = action.ljust(6)  # Pad "buy" or "sell" to 6 characters
    #    file_logger.info(f"Action: {action_aligned} | Crypto Price USD: {usd_non_sol_price:.8f} | Wallet Value USD: {usd_wallet_value_before:.8f} | Before Transaction | Attempt: {attempt}")
    file_logger.info(f"Action: {action_aligned} | mint: {DESIRED_CRYPTO_TICKER} | Before Transaction | Attempt: {attempt} | Order Size: {order_size}")

    prioritizationFeeLamports = 1000000

    # Construct the payload with dynamic outAmount
    payload = json.dumps({
        "quoteResponse": quote_response,
        "userPublicKey": WALLET_ID,
        "wrapAndUnwrapSol": True,
        "dynamicComputeUnitLimit": True,
        "dynamicSlippage": True,
        "prioritizationFeeLamports": prioritizationFeeLamports
    })

    #2000000

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {JUPITER_API_KEY}'
    }
    url = "https://quote-api.jup.ag/v6/swap"

    try:
        logger.info(f"Sending swap request to Jupiter API: {url}")
        response = requests.post(url, headers=headers, data=payload)
        logger.info(response)

        if response.status_code == 200:
            response_data = response.json()
            logger.info(f"Swap response: {response_data}")

            swap_transaction = response.json().get("swapTransaction")
            logger.info(f"Swap successful: {swap_transaction}")

            # Call sign_and_send_transaction and capture its response
            txn_response = await sign_and_send_transaction(swap_transaction)

            # Check if the response indicates success or failure and handle accordingly
            if txn_response.get("success"):
                logger.info(f"Transaction successful: {txn_response['transaction_id']}")

                #                sol_balance_new = get_native_sol_balance(WALLET_ID)
                #                crypto_balance_new = get_wallet_balance(WALLET_ID, DESIRED_CRYPTO_TICKER)
                #
                #                # Calculate USD value of wallet after transaction
                #                usd_wallet_value_after = (usd_sol_price * sol_balance_new) + (usd_non_sol_price * crypto_balance_new)
                #                file_logger.info(f"Action: {action_aligned} | Crypto Price USD: {usd_non_sol_price:.8f} | Wallet Value USD: {usd_wallet_value_after:.8f} | After Transaction")
                file_logger.info(f"Action: {action_aligned} | mint: {DESIRED_CRYPTO_TICKER} | After Transaction")

                sol_amount = get_sol_amount(quote_response)  # Your function to get sol amount
                prioritization_fee_sol = 0.001  # Example value, adjust as needed

                # Set txn_amount_sol based on action (buy or sell)
                if action == "buy":
                    txn_amount_sol = sol_amount + prioritization_fee_sol
                elif action == "sell":
                    txn_amount_sol = sol_amount - prioritization_fee_sol
                else:
                    txn_amount_sol = 0  # Default value in case of invalid action (optional)

                logger.info(f"sol_amount: {sol_amount}")

                # Log the transaction amount for the action
                transaction_logger.info(f"Action: {action_aligned} | mint: {DESIRED_CRYPTO_TICKER} | txn_amount_sol: {txn_amount_sol:.5f} | percent: {percentage}")

                return {"status": "success", "transaction_id": txn_response['transaction_id']}
            else:
                logger.error(f"Transaction failed: {txn_response.get('error')}")
                return {"status": "error", "message": txn_response.get("error"), "details": txn_response.get("details")}

        else:
            logger.error(f"Swap failed with status code {response.status_code}: {response.text}")
            return {"status": "error", "message": "Swap failed", "details": response.json()}
    except Exception as e:
        logger.error(f"Error during swap request: {str(e)}")
        return {"status": "error", "message": "Swap request failed", "details": str(e)}

def calculate_amounts(
        holding_amount,
        current_price,
        input_token_decimals,
        output_token_decimals,
        increase_percent,
        profit_percent,
):
    """
    Calculates makingAmount and takingAmount based on a specified price increase percentage and profit percentage.

    :param holding_amount: Total amount of input tokens held (in smallest unit, e.g., lamports for SOL).
    :param current_price: Current price of the input token (e.g., in USD).
    :param input_token_decimals: Decimals for the input token (e.g., 9 for SOL).
    :param output_token_decimals: Decimals for the output token (e.g., 6 for USDC).
    :param increase_percent: Percentage price increase to trigger the sale (e.g., 250 for 250% increase).
    :param profit_percent: Percentage of the profit to sell (e.g., 50 for 50% profit).
    :return: makingAmount and takingAmount.
    """
    # Calculate the new price after the specified percentage increase
    new_price = current_price * (1 + increase_percent / 100)

    # Total value at the current price
    original_value = holding_amount * (current_price / (10 ** input_token_decimals))

    # New value when the price increases
    new_value = holding_amount * (new_price / (10 ** input_token_decimals))

    # Profit calculation
    profit = new_value - original_value

    # Sell the specified percentage of the profit
    making_amount = (profit_percent / 100) * profit / (new_price / (10 ** input_token_decimals))
    taking_amount = (profit_percent / 100) * profit * (10 ** output_token_decimals)

    return int(making_amount), int(taking_amount)

def extract_price_from_quote(quote_response):
    # Extract input and output amounts
    input_amount = int(quote_response["inAmount"])  # Input amount in lamports
    output_amount = int(quote_response["outAmount"])  # Output amount in smallest token unit

    # Convert input to SOL (1 SOL = 1,000,000,000 lamports)
    input_amount_sol = input_amount / 1000000000

    # Assume output_amount is already in native units (e.g., USDC or another token)
    # To calculate the price of the output token in terms of SOL:
    price_in_sol = output_amount / input_amount_sol

    return price_in_sol

# Function to perform swap
# @current_app.task
async def create_limit_order(action, attempt, increase_percent, profit_percent):
    logger.info(f"create limit order: {action}")

    EXPIRATION = int(time.time()) + (30 * 24 * 60 * 60)  # 30 days in seconds
    COMPUTE_UNIT_PRICE = "auto"

    # Get balances
    # sol_balance = get_native_sol_balance(WALLET_ID)
    # crypto_balance = get_wallet_balance(WALLET_ID, DESIRED_CRYPTO_TICKER)

    order_size = "full"

    if action == "buy":
        # sol_balance = get_native_sol_balance(WALLET_ID)
        # # Determine the amount to swap
        # if sol_balance > 0.5:
        #     amount_to_swap = SWAP_AMOUNT  # Use 0.5 SOL if balance is greater than 0.5
        # else:
        #     amount_to_swap = sol_balance * 0.25  # Otherwise, use 50% of the balance
        # amount_to_swap = percentage
        #
        # order_size = f"{amount_to_swap:.2f} SOL"
        #
        # logger.info(f"Amount to swap (buy): {order_size}")
        amount_to_swap = .05
        input_mint = SOL_TICKER
        output_mint = DESIRED_CRYPTO_TICKER
    elif action == "sell":
        crypto_balance = get_wallet_balance(WALLET_ID, DESIRED_CRYPTO_TICKER)

        if crypto_balance == 0:
            logger.error(f"{DESIRED_CRYPTO_TICKER} balance is 0. Cannot perform sell action.")
            return {"error": f"Insufficient {DESIRED_CRYPTO_TICKER} balance to perform sell action."}

        # # Determine the amount to swap for selling
        # if percentage is not None:
        #     amount_to_swap = crypto_balance * percentage  # Use specified percentage
        #     if percentage < 1.00:
        #         order_size = 'partial'
        # else:
        amount_to_swap = crypto_balance * profit_percent/100  # Default to 100%

        # logger.info(f"Amount to swap (sell): {amount_to_swap} {DESIRED_CRYPTO_TICKER}")
        input_mint = DESIRED_CRYPTO_TICKER
        output_mint = SOL_TICKER
    else:
        logger.error(f"Invalid action: {action}")
        return {"error": "Invalid action specified"}

    action_aligned = action.ljust(6)  # Pad "buy" or "sell" to 6 characters
    #    file_logger.info(f"Action: {action_aligned} | Crypto Price USD: {usd_non_sol_price:.8f} | Wallet Value USD: {usd_wallet_value_before:.8f} | Before Transaction | Attempt: {attempt}")
    file_logger.info(f"Action: {action_aligned} | mint: {DESIRED_CRYPTO_TICKER} | Before Transaction | Attempt: {attempt} | Order Size: {order_size}")

    # Fetch decimals for the input token
    input_token_decimals = get_token_decimals(input_mint)

    # Fetch decimals for the output token
    # output_token_decimals = get_token_decimals(output_mint)
    output_token_decimals = 9

    # Example usage:
    quote_response = get_swap_quote_response(input_mint, amount_to_swap, output_mint, 0, input_token_decimals)

    # price_in_sol = extract_price_from_quote(quote_response)

    # Extract the crypto amounts
    crypto_count = get_crypto_count(quote_response, input_token_decimals)

    # Get Solana's USD price
    usd_sol_price = get_usd_price("solana")

    # Calculate USD price of the non-Solana token
    if crypto_count.get("sol") and crypto_count.get("non_sol"):
        # Convert input and output amounts to token's natural units
        input_amount_normalized = crypto_count["non_sol"]
        output_amount_normalized = crypto_count["sol"]

        # Calculate the price of the input token in SOL
        price_in_sol = output_amount_normalized / input_amount_normalized

        usd_non_sol_price = (usd_sol_price * crypto_count["sol"]) / crypto_count["non_sol"]
        logger.info(f"USD price of the non-Solana token: {usd_non_sol_price}")
    else:
        logger.info("Error: Missing Solana or non-Solana amounts in the swap quote.")


    making_amount = crypto_count["non_sol_lamports"]
    taking_amount = int(crypto_count["sol_lamports"] * (1 + increase_percent / 100))

    logger.info(f"Input Token Decimals: {input_token_decimals}")
    logger.info(f"Output Token Decimals: {output_token_decimals}")
    logger.info(f"Quote Response: {quote_response}")
    logger.info(f"Price Extracted SOL: {price_in_sol}")
    logger.info(f"Price Extracted USD: {usd_non_sol_price}")
    logger.info(f"Crypto Balance: {crypto_balance}")


    logger.info(f"Making Amount (input token): {making_amount}")
    logger.info(f"Taking Amount (output token): {taking_amount}")

    """Creates a limit order on Jupiter."""

    payload = {
        "inputMint": str(input_mint),
        "outputMint": str(output_mint),
        "maker": str(WALLET_ID),
        "payer": str(WALLET_ID),
        "params": {
            "makingAmount": str(making_amount),
            "takingAmount": str(taking_amount),
            "expiredAt": str(EXPIRATION),
        },
        "computeUnitPrice": COMPUTE_UNIT_PRICE,
    }

    logger.info(f"Payload: {json.dumps(payload, indent=4)}")

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {JUPITER_API_KEY}'
    }

    url = "https://api.jup.ag/limit/v2/createOrder"

    try:
        # Send the request to Jupiter
        response = requests.post(url, headers=headers, json=payload)

        # Log the response
        logger.info(f"Response Status Code: {response.status_code}")
        logger.info(f"Response Body: {response.text}")

        # If response is successful, you can parse the response as JSON
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Response JSON: {json.dumps(result, indent=4)}")
            tx = result["tx"]
            logger.info(tx)
            # Call sign_and_send_transaction and capture its response
            txn_response = await sign_and_send_transaction(tx)

            # Check if the response indicates success or failure and handle accordingly
            if txn_response.get("success"):
                logger.info(f"Transaction successful: {txn_response['transaction_id']}")

                #                sol_balance_new = get_native_sol_balance(WALLET_ID)
                #                crypto_balance_new = get_wallet_balance(WALLET_ID, DESIRED_CRYPTO_TICKER)
                #
                #                # Calculate USD value of wallet after transaction
                #                usd_wallet_value_after = (usd_sol_price * sol_balance_new) + (usd_non_sol_price * crypto_balance_new)
                #                file_logger.info(f"Action: {action_aligned} | Crypto Price USD: {usd_non_sol_price:.8f} | Wallet Value USD: {usd_wallet_value_after:.8f} | After Transaction")
                file_logger.info(f"Action: {action_aligned} | mint: {DESIRED_CRYPTO_TICKER} | After Transaction")

                sol_amount = get_sol_amount(quote_response)  # Your function to get sol amount
                prioritization_fee_sol = 0.0001  # Example value, adjust as needed

                # Set txn_amount_sol based on action (buy or sell)
                if action == "buy":
                    txn_amount_sol = sol_amount + prioritization_fee_sol
                elif action == "sell":
                    txn_amount_sol = sol_amount - prioritization_fee_sol
                else:
                    txn_amount_sol = 0  # Default value in case of invalid action (optional)

                logger.info(f"sol_amount: {sol_amount}")

                # Log the transaction amount for the action
                transaction_logger.info(f"Action: {action_aligned} | mint: {DESIRED_CRYPTO_TICKER} | txn_amount_sol: {txn_amount_sol:.5f} | percent: {profit_percent}")

                return {"status": "success", "transaction_id": txn_response['transaction_id']}
            else:
                logger.error(f"Transaction failed: {txn_response.get('error')}")
                return {"status": "error", "message": txn_response.get("error"), "details": txn_response.get("details")}

        else:
            logger.error(f"Error creating limit order: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while making the request: {e}")
        return None

    #2000000

    # url = "https://quote-api.jup.ag/v6/swap"
    #
    # try:
    #     logger.info(f"Sending swap request to Jupiter API: {url}")
    #     response = requests.post(url, headers=headers, data=payload)
    #     logger.info(response)
    #
    # logger.info(result)
    # # Deserialize the transaction
    # tx = result["tx"]
    # logger.info(tx)

    # # Construct the payload with dynamic outAmount
    # payload = json.dumps({
    #     "quoteResponse": quote_response,
    #     "userPublicKey": WALLET_ID,
    #     "wrapAndUnwrapSol": True,
    #     "dynamicComputeUnitLimit": True,
    #     "dynamicSlippage": True,
    #     "prioritizationFeeLamports": prioritizationFeeLamports
    # })
    #
    # #2000000
    #
    # headers = {
    #     'Content-Type': 'application/json',
    #     'Accept': 'application/json',
    #     'Authorization': f'Bearer {JUPITER_API_KEY}'
    # }
    # url = "https://quote-api.jup.ag/v6/swap"

    # try:
    #     logger.info(f"Sending swap request to Jupiter API: {url}")
    #     response = requests.post(url, headers=headers, data=payload)
    #     logger.info(response)
    #
    #     if response.status_code == 200:
    #         response_data = response.json()
    #         logger.info(f"Swap response: {response_data}")
    #
    #         swap_transaction = response.json().get("swapTransaction")
    #         logger.info(f"Swap successful: {swap_transaction}")
    #
    #         # Call sign_and_send_transaction and capture its response
    #         txn_response = await sign_and_send_transaction(swap_transaction)
    #
    #         # Check if the response indicates success or failure and handle accordingly
    #         if txn_response.get("success"):
    #             logger.info(f"Transaction successful: {txn_response['transaction_id']}")
    #
    #             #                sol_balance_new = get_native_sol_balance(WALLET_ID)
    #             #                crypto_balance_new = get_wallet_balance(WALLET_ID, DESIRED_CRYPTO_TICKER)
    #             #
    #             #                # Calculate USD value of wallet after transaction
    #             #                usd_wallet_value_after = (usd_sol_price * sol_balance_new) + (usd_non_sol_price * crypto_balance_new)
    #             #                file_logger.info(f"Action: {action_aligned} | Crypto Price USD: {usd_non_sol_price:.8f} | Wallet Value USD: {usd_wallet_value_after:.8f} | After Transaction")
    #             file_logger.info(f"Action: {action_aligned} | mint: {DESIRED_CRYPTO_TICKER} | After Transaction")
    #
    #             sol_amount = get_sol_amount(quote_response)  # Your function to get sol amount
    #             prioritization_fee_sol = 0.001  # Example value, adjust as needed
    #
    #             # Set txn_amount_sol based on action (buy or sell)
    #             if action == "buy":
    #                 txn_amount_sol = sol_amount + prioritization_fee_sol
    #             elif action == "sell":
    #                 txn_amount_sol = sol_amount - prioritization_fee_sol
    #             else:
    #                 txn_amount_sol = 0  # Default value in case of invalid action (optional)
    #
    #             logger.info(f"sol_amount: {sol_amount}")
    #
    #             # Log the transaction amount for the action
    #             transaction_logger.info(f"Action: {action_aligned} | mint: {DESIRED_CRYPTO_TICKER} | txn_amount_sol: {txn_amount_sol:.5f} | percent: {percentage}")
    #
    #             return {"status": "success", "transaction_id": txn_response['transaction_id']}
    #         else:
    #             logger.error(f"Transaction failed: {txn_response.get('error')}")
    #             return {"status": "error", "message": txn_response.get("error"), "details": txn_response.get("details")}
    #
    #     else:
    #         logger.error(f"Swap failed with status code {response.status_code}: {response.text}")
    #         return {"status": "error", "message": "Swap failed", "details": response.json()}
    # except Exception as e:
    #     logger.error(f"Error during swap request: {str(e)}")
    #     return {"status": "error", "message": "Swap request failed", "details": str(e)}

def create_associated_token_account(payer: Pubkey, owner: Pubkey, mint: Pubkey) -> TransactionInstruction:
    logger.info(f"step: {1}")
    # Create the instruction without deriving the associated token program ID
    instruction = Instruction(
        program_id=TOKEN_PROGRAM_ID,  # Use the token program ID here
        accounts=[
            AccountMeta(pubkey=payer, is_signer=True, is_writable=True),
            AccountMeta(pubkey=associated_token_address, is_signer=False, is_writable=True),
            AccountMeta(pubkey=owner, is_signer=False, is_writable=False),
            AccountMeta(pubkey=mint, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYS_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=TOKEN_PROGRAM_ID, is_signer=False, is_writable=False),
            AccountMeta(pubkey=SYSVAR_RENT_PUBKEY, is_signer=False, is_writable=False),
        ],
        data=bytes([]),  # Associated token account instructions don't need extra data
    )

# Function to sign and send the transaction with retries
async def sign_and_send_transaction(swap_transaction):
    try:
        # Log incoming base64-encoded transaction for debugging
        logger.info(f"Base64-encoded swap transaction: {swap_transaction}")

        # Validate base64 encoding
        base64_pattern = re.compile('^[A-Za-z0-9+/=]+$')
        if not base64_pattern.match(swap_transaction):
            raise ValueError("Invalid base64 string")

        # Decode base64-encoded transaction to raw bytes
        try:
            raw_transaction = VersionedTransaction.from_bytes(base64.b64decode(swap_transaction))
        except binascii.Error as e:
            logger.error(f"Base64 decoding error: {str(e)}")
            return {"error": "Decoding error", "details": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": "Unexpected decoding error", "details": str(e)}

        # Log the decoded transaction as bytes and hex
        logger.info(f"Decoded raw transaction bytes: {raw_transaction}")

        # mint = Pubkey.from_string(DESIRED_CRYPTO_TICKER)
        # Create the associated token account instruction
        # ata_instruction = create_associated_token_account(payer, owner, mint)

        # logger.info(f"ata_instruction: {ata_instruction}")

        # Add the instruction to the transaction
        # raw_transaction.message.instructions.insert(0, ata_instruction)

        # Log the updated transaction with ATA instruction
        # logger.info(f"Transaction with ATA instruction: {raw_transaction}")

        # Decode and sign the transaction
        signature = PRIVATE_KEY_PAIR.sign_message(message.to_bytes_versioned(raw_transaction.message))
        logger.info(f"Signature: {signature}")

        signed_txn = VersionedTransaction.populate(raw_transaction.message, [signature])
        logger.info(f"Signed transaction: {signed_txn}")

        # Set transaction options
        opts = TxOpts(skip_preflight=False, preflight_commitment=Processed)

        # Retry logic for sending the transaction
        max_retries = 3
        attempt = 1
        transaction_id = None

        while attempt <= max_retries:
            try:
                # Send the signed transaction
                response = client.send_raw_transaction(txn=bytes(signed_txn), opts=opts)
                transaction_id = str(response.value) if hasattr(response, 'value') else None

                # Check if the transaction was successful
                if transaction_id:
                    logger.info(f"Transaction sent successfully: https://explorer.solana.com/tx/{transaction_id}")
                    break  # Exit the loop on success
                else:
                    logger.warning(f"Transaction attempt {attempt} failed with response: {response}")

            except Exception as e:
                logger.error(f"Transaction attempt {attempt} encountered an error: {str(e)}")

            # Increment attempt and wait before retrying with exponential backoff
            attempt += 1
            if attempt <= max_retries:
                wait_time = 2 ** (attempt - 1)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        # Final check if the transaction was successful
        if not transaction_id:
            return {"error": "Transaction failed after multiple attempts"}

        try:
            solana_rpc_url = "https://api.mainnet-beta.solana.com"

            # Create an AsyncClient instance
            connection = AsyncClient(solana_rpc_url)

            # Wait for the transaction to be finalized
            transaction_status = await get_transaction_status(connection, transaction_id)

            # Close the connection
            await connection.close()

            if transaction_status['status'] == "finalized":
                logger.info(f"Transaction {transaction_id} finalized successfully.")

                return {"success": True, "transaction_id": transaction_id}
            elif transaction_status['status'] == "Error":
                logger.info(f"Transaction {transaction_id} Error.")
                return {"error": "Transaction Status contains error"}


            # max_attempts = 3  # max retries (4 attempts in 2 minutes)
            # wait_time = 30    # wait time between each check in seconds
            # attempt = 1
            #
            # while attempt <= max_attempts:
            #     # Wait before the next check
            #     time.sleep(wait_time)
            #
            #     # Check the transaction status
            #     transaction_status = get_transaction_status(transaction_id)
            #     if transaction_status['status'] == "finalized":
            #         logger.info(f"Transaction {transaction_id} finalized successfully.")
            #
            #
            #         return {"success": True, "transaction_id": transaction_id}
            #     elif transaction_status['status'] == "Error":
            #         logger.info(f"Transaction {transaction_id} Error.")
            #         return {"error": "Transaction Status contains error"}
            #
            #     # Increment attempt count
            #     attempt += 1
            #
            # # If loop exits without finalizing
            # logger.error(f"Transaction {transaction_id} did not finalize within allowed time.")
            # return {"error": "Transaction did not finalize within 2 minutes"}

        except Exception as e:
            logger.error(f"Error while validating transaction finalization: {e}")
            return {"error": "Transaction validation failed", "details": str(e)}

    except Exception as e:
        logger.error(f"Error during transaction signing and sending: {e}")
        return {"error": "Transaction signing and sending failed", "details": str(e)}

async def wait_for_finalized_status(connection, tx_signature, polling_interval=5, timeout=300):
    """
    Waits until a transaction reaches 'finalized' status and returns success or error.

    Args:
        connection (AsyncClient): An instance of Solana AsyncClient.
        tx_signature (str): The transaction signature.
        polling_interval (int): Time in seconds between each status check.
        timeout (int): Maximum time in seconds to wait for 'finalized' status.

    Returns:
        str: "success" if the transaction finalized without errors, or "error" otherwise.
    """
    elapsed_time = 0
    while elapsed_time < timeout:
        try:
            signature_obj = Signature.from_string(tx_signature)
            # Fetch the status
            result = await connection.get_signature_statuses([signature_obj])
            status = result["result"]["value"][0]

            if status is not None:
                confirmation_status = status.get("confirmationStatus", None)
                logger.info("Current confirmation status: %s", confirmation_status)

                # Check for finalization
                if confirmation_status == "finalized":
                    logger.info("Transaction %s is finalized.", tx_signature)

                    # Check for errors in the transaction
                    if status.get("err") is None:
                        logger.info("Transaction %s finalized successfully.", tx_signature)
                        return "success"
                    else:
                        logger.error("Transaction %s failed with error: %s", tx_signature, status["err"])
                        return "error"
            else:
                logger.warning("Transaction %s not found. Retrying...", tx_signature)

        except Exception as e:
            logger.error("Error fetching confirmation status: %s", e)
            return "error"

        # Wait before retrying
        await asyncio.sleep(polling_interval)
        elapsed_time += polling_interval

    logger.error("Timeout reached while waiting for transaction %s to be finalized.", tx_signature)
    return "error"

async def listen_for_finalization(transaction_signature):
    """
    Listens for the finalization of a Solana transaction.

    Args:
        transaction_signature (str): The signature of the transaction to monitor.

    Returns:
        str: "success" if the transaction finalized successfully, or "error" if it failed.
    """
    url = "wss://api.mainnet-beta.solana.com"
    try:
        async with websockets.connect(url) as websocket:
            # Subscribe to the transaction
            subscribe_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "signatureSubscribe",
                "params": [
                    transaction_signature,
                    {"commitment": "finalized"}
                ]
            }
            await websocket.send(json.dumps(subscribe_message))
            logger.info("Subscribed to transaction: %s", transaction_signature)

            # Wait for the response
            while True:
                response = await websocket.recv()
                response_data = json.loads(response)
                logger.info("Received response: %s", response_data)

                # Check for finalization or errors
                if response_data.get("method") == "signatureNotification":
                    result = response_data["params"]["result"]
                    if result.get("err") is None:
                        logger.info("Transaction finalized successfully!")
                        return "success"
                    else:
                        logger.error("Transaction failed with error: %s", result["err"])
                        return "error"
    except Exception as e:
        logger.exception("An error occurred while listening for transaction finalization: %s", e)
        return "error"

async def subscribe_transaction_status(txid):
    async with connect("wss://empty-delicate-aura.solana-mainnet.quiknode.pro/eda79e548ae61382c63ac4e795fdd096ef2f24dd") as websocket:
        logger.info("Connected to QuickNode WebSocket")
        signature_obj = Signature.from_string(txid)
        logger.info(f"Signature object: {signature_obj}")

        # Subscribe with commitment level set to finalized
        await websocket.signature_subscribe(signature_obj, commitment="finalized")

        # Log the initial response
        first_resp = await websocket.recv()
        logger.info(f"First response: {first_resp}")

        if isinstance(first_resp, list) and len(first_resp) > 0:
            subscription_id = first_resp[0].result
            logger.info(f"Subscription ID: {subscription_id}")
        else:
            logger.error("Unexpected format for first_resp. Cannot find subscription ID.")
            return

        logger.info("Entering message processing loop...")

        try:
            async for idx, msg in enumerate(websocket):
                logger.info(f"Message {idx}: {msg}")
                # Check if the message contains finalized status
                if "finalized" in str(msg):
                    logger.info(f"Transaction finalized: {msg}")
                    break  # Exit the loop once the transaction is finalized
                else:
                    logger.info("Transaction is not finalized yet. Waiting for updates...")
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for transaction status updates.")

# Function to fetch transaction status and log the response
async def get_transaction_status(connection, txid, polling_interval=5, timeout=300):
    elapsed_time = 0
    while elapsed_time < timeout:
        try:
            signature_obj = Signature.from_string(txid)
            logger.info(f"Signature object: {signature_obj}")

            response = await connection.get_signature_statuses([signature_obj])

            # response = client.get_signature_statuses([signature_obj])

            logger.info(f"Transaction status response: {response}")

            # Access the value directly from response
            if response and response.value:
                # Extract the transaction's status details
                status_info = response.value[0]  # response.value is a list of status objects

                if status_info is not None:
                    confirmation_status = str(status_info.confirmation_status)  # Directly access confirmation_status attribute
                    logger.info(f"Transaction {txid} confirmation_status: {confirmation_status}")

                    error_status = str(status_info.err)
                    logger.info(f"Transaction {txid} err: {error_status}")

                    if error_status != "None":
                        logger.info(f"Transaction {txid} contains an Error")
                        return {"status": "Error", "txid": txid}

                    # Log the confirmation status and return if "finalized"
                    if confirmation_status == "TransactionConfirmationStatus.Finalized":
                        logger.info(f"Transaction {txid} has been finalized.")
                        return {"status": "finalized", "txid": txid}
                    else:
                        logger.info(f"Transaction {txid} status: {confirmation_status}")
                        # return {"status": confirmation_status, "txid": txid}
                else:
                    logger.warning(f"Transaction {txid} not found or still pending.")
                    # return {"status": "pending", "txid": txid}
            else:
                logger.warning(f"No status found for transaction {txid}")
                # return {"status": "no_status", "txid": txid}
        except Exception as e:
            logger.error(f"Error fetching transaction status for {txid}: {str(e)}")
            # return {"status": "error", "txid": txid, "error_message": str(e)}

        # Wait before retrying
        await asyncio.sleep(polling_interval)
        elapsed_time += polling_interval

    logger.error("Timeout reached while waiting for transaction %s to be finalized.", tx_signature)
    return "error"


@app.route('/webhook', methods=['POST'])
async def webhook_listener():
    global buy_task, sell_task, DESIRED_CRYPTO_TICKER

    # Parse JSON data from the request
    data = await request.get_json()

    # Extract the fields from the data
    action = data.get("action")
    mint = data.get("mint")
    percent = data.get("percent")

    action_aligned = action.ljust(6)  # Pad "buy" or "sell" to 6 characters

    # Log the received data for debugging
    logger.info(f"Action: {action_aligned} | mint: {mint} | Percent: {percent}")


    # Acquire the lock to ensure mutual exclusion on state changes
    async with action_lock:
        DESIRED_CRYPTO_TICKER = mint  # Replace with actual ticker ID
        file_logger.info(f"Action: {action_aligned} | mint: {mint} | Received Alert Price | Percent: {percent}")

        try:
            if action == "buy":
                # Create and await the buy task
                buy_task = asyncio.create_task(handle_buy(action, percent))
                await buy_task

            elif action == "sell":
                # Create and await the sell task
                sell_task = asyncio.create_task(handle_sell(action, percent))
                await sell_task

            else:
                result = {"status": "error", "message": "Invalid action"}
                logger.warning(f"Invalid action received: {action}")
                return jsonify(result)

        except Exception as e:
            # Log and handle any exceptions that occur during the action
            error_message = f"Exception during {action} action for {mint}: {str(e)}"
            logger.error(error_message)
            result = {"status": "error", "message": error_message}
            restart_server()
        else:
            # If no exception, indicate success
            result = {"status": "success", "message": f"{action} action completed successfully"}

    # Log the result after processing the action
    logger.info(f"Processed action: {action}, result: {result}")

    return jsonify(result)

async def handle_buy(action, percent):

    slippage_percentage = 3.0
    max_retries = RETRY_COUNT
    attempt = 1
    success = False


    while attempt <= max_retries and not success:
        logger.info(f"Attempt {attempt} for {action} swap with {slippage_percentage}% slippage")

        result = await perform_swap(action, slippage_percentage, attempt, percent)

        # result = await create_limit_order("sell", attempt, 100, 50)
        result = await create_limit_order("sell", attempt, 200, 33)
        result = await create_limit_order("sell", attempt, 500, 50)
        # result = await create_limit_order("sell", attempt, 500, 40)
        # result = await create_limit_order("sell", attempt, 1000, 40)
        # result = await create_limit_order("sell", attempt, 2000, 40)
        # result = await create_limit_order("sell", attempt, 4000, 40)
        # result = await create_limit_order("sell", attempt, 8000, 40)
        # result = await create_limit_order("sell", attempt, 10000, 100)

        if result and result['status'] == "success":
            success = True
            logger.info("Buy swap succeeded")
        else:
            logger.warning(f"Attempt {attempt} for {action} failed, increasing slippage to {slippage_percentage}%")
            slippage_percentage += 0.5
            attempt += 1

    if not success:
        logger.error("Buy swap failed after maximum retries")

        result = {"status": "error", "message": "Buy swap failed after maximum retries"}
    else:
        result = {"status": "success", "message": "Buy swap succeeded"}

    return result

async def handle_sell(action, percent):
    global buy_active, sell_active, buy_task, sell_task

    slippage_percentage = 1.0
    max_retries = RETRY_COUNT + 0
    attempt = 1
    success = False

    while attempt <= max_retries and not success:
        logger.info(f"Attempt {attempt} for {action} swap with {slippage_percentage}% slippage")

        # result = await perform_swap(action, slippage_percentage, attempt, percent)
        result = await create_limit_order(action, attempt, 10, 50)

        # Check if the result indicates insufficient balance
        if 'error' in result and 'Insufficient' in result['error']:
            logger.error(f"Swap failed: {result['error']}. Not retrying swap.")
            # Format aligned action and mint
            action_aligned = action.ljust(6)  # Pad "buy" or "sell" to 6 characters
            file_logger.info(
                f"Action: {action_aligned} | mint: {DESIRED_CRYPTO_TICKER} | Transaction Skipped: Insufficient Balance"
            )
            break

        if result and result['status'] == "success":
            success = True
            logger.info("Sell swap succeeded")
        else:
            logger.warning(f"Attempt {attempt} for {action} failed, increasing slippage to {slippage_percentage}%")
            slippage_percentage += 0.5
            attempt += 1

    if not success:
        logger.error("Sell swap failed after maximum retries")
        result = {"status": "error", "message": "Sell swap failed after maximum retries"}
    else:
        result = {"status": "success", "message": "Sell swap succeeded"}

    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
