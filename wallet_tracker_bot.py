from solana.rpc.api import Client
# from solana.publickey import Pubkey  # Import Pubkey to convert the address
import time
from solders.signature import Signature
from solders.pubkey import Pubkey
from solders.transaction_status import EncodedConfirmedTransactionWithStatusMeta
import json
import re

# Step 2: Connect to the Solana blockchain
solana_client = Client("https://api.mainnet-beta.solana.com")  # Use mainnet or devnet/testnet if needed

SOL_MINT = "So11111111111111111111111111111111111111112"  # Solana's mint address

# Function to extract the first SOL and non-SOL mint balances
# Extract most sol balance
def extract_first_sol_and_non_sol_balances(token_balances):
    sol_balance = None
    non_sol_balance = None

    print("Starting extraction of SOL and non-SOL balances...")

    for balance in token_balances:
        mint = balance.get("mint")
        ui_token_amount = balance.get("uiTokenAmount", {})

        print(f"Processing balance: {balance}")

        # Check if uiAmount is available for valid SOL and non-SOL balances
        if ui_token_amount and ui_token_amount.get("uiAmount") is not None:
            print(f"Found valid uiTokenAmount: {ui_token_amount}")

            # Extract first SOL balance
            if mint == "So11111111111111111111111111111111111111112" and sol_balance is None:
                print(f"Found first SOL balance: {balance}")
                sol_balance = {
                    "accountIndex": balance.get("accountIndex"),
                    "mint": mint,
                    "uiTokenAmount": ui_token_amount
                }

            # Extract first non-SOL balance
            elif mint != "So11111111111111111111111111111111111111112" and non_sol_balance is None:
                print(f"Found first non-SOL balance: {balance}")
                non_sol_balance = {
                    "accountIndex": balance.get("accountIndex"),
                    "mint": mint,
                    "uiTokenAmount": ui_token_amount
                }

        # If both SOL and non-SOL balances are found, break the loop
        if sol_balance and non_sol_balance:
            print("Both SOL and non-SOL balances found. Exiting loop.")
            break

    print(f"Extracted SOL balance: {sol_balance}")
    print(f"Extracted non-SOL balance: {non_sol_balance}")

    return sol_balance, non_sol_balance

def extract_largest_sol_and_non_sol_balances(token_balances):
    sol_balance = None
    non_sol_balance = None

    print("Starting extraction of the largest SOL and non-SOL balances...")

    for balance in token_balances:
        mint = balance.get("mint")
        ui_token_amount = balance.get("uiTokenAmount", {})

        print(f"Processing balance: {balance}")

        # Check if uiAmount is available for valid SOL and non-SOL balances
        if ui_token_amount and ui_token_amount.get("uiAmount") is not None:
            print(f"Found valid uiTokenAmount: {ui_token_amount}")

            # Check and update SOL balance if it's larger
            if mint == "So11111111111111111111111111111111111111112":
                if sol_balance is None or ui_token_amount["uiAmount"] > sol_balance["uiTokenAmount"]["uiAmount"]:
                    print(f"Updating largest SOL balance: {balance}")
                    sol_balance = {
                        "accountIndex": balance.get("accountIndex"),
                        "mint": mint,
                        "uiTokenAmount": ui_token_amount
                    }

            # Check and update non-SOL balance if it's larger
            else:
                if non_sol_balance is None or ui_token_amount["uiAmount"] > non_sol_balance["uiTokenAmount"]["uiAmount"]:
                    print(f"Updating largest non-SOL balance: {balance}")
                    non_sol_balance = {
                        "accountIndex": balance.get("accountIndex"),
                        "mint": mint,
                        "uiTokenAmount": ui_token_amount
                    }

    print(f"Extracted largest SOL balance: {sol_balance}")
    print(f"Extracted largest non-SOL balance: {non_sol_balance}")

    return sol_balance, non_sol_balance


def extract_mints2(transaction_json):
    # Convert JSON to a string for regex search
    json_str = json.dumps(transaction_json)

    # Regex pattern to match mint addresses
    mint_pattern = r'"mint":\s*"([1-9A-HJ-NP-Za-km-z]{32,44})"|mint:\s*"([1-9A-HJ-NP-Za-km-z]{32,44})"'

    # Find all mint addresses in the JSON string
    mint_matches = re.findall(mint_pattern, json_str)

    # Extract only non-empty matches from both capture groups
    mints = [m for match in mint_matches for m in match if m]

    # Find valid input/output mint pairs
    input_mint, output_mint = None, None
    for i in range(len(mints) - 1):
        if mints[i] != mints[i + 1]:  # Ensure different mints
            input_mint, output_mint = mints[i], mints[i + 1]
            break  # Stop searching once a valid pair is found

    if input_mint and output_mint:
        return input_mint, output_mint
    else:
        return None, None


def extract_mints(transaction):
    pre_mint_pattern = re.compile(r'pre_token_balances: Some\(\[UiTransactionTokenBalance \{ account_index: \d+, mint: "([^"]+)"')
    post_mint_pattern = re.compile(r'post_token_balances: Some\(\[UiTransactionTokenBalance \{ account_index: \d+, mint: "([^"]+)"')

    pre_mints = pre_mint_pattern.findall(transaction)
    post_mints = post_mint_pattern.findall(transaction)

    for in_mint in pre_mints:
        for out_mint in post_mints:
            if in_mint != out_mint:
                return in_mint, out_mint  # Return the first valid pair found

    return None, None  # No valid pair found

def analyze_transaction(transaction):
    try:
        action = "none"
        # Convert Solders object to dictionary
        transaction_dict = json.loads(transaction.to_json())


        if "meta" not in transaction_dict:
            return {"error": "Transaction metadata is missing"}


        # Extract pre and post token balances
        pre_token_balances = transaction_dict.get("meta", {}).get("preTokenBalances", [])
        post_token_balances = transaction_dict.get("meta", {}).get("postTokenBalances", [])
        inner_instructions = transaction_dict.get("meta", {}).get("innerInstructions", [])

        non_sol_mint = None

        print(pre_token_balances)
        print(post_token_balances)

        pre_sol_balance, pre_non_sol_balance = extract_largest_sol_and_non_sol_balances(pre_token_balances)
        post_sol_balance, post_non_sol_balance = extract_largest_sol_and_non_sol_balances(post_token_balances)

        print(inner_instructions)


        # Assign the first non-SOL mint to non_sol_mint (if available)
        if pre_non_sol_balance:
            non_sol_mint = pre_non_sol_balance.get("mint")
        elif post_non_sol_balance:
            non_sol_mint = post_non_sol_balance.get("mint")

        # Check if SOL balance has increased or decreased (buy or sell)
        if pre_sol_balance and post_sol_balance:
            pre_sol_amount = int(pre_sol_balance["uiTokenAmount"]["amount"])
            post_sol_amount = int(post_sol_balance["uiTokenAmount"]["amount"])

            if post_sol_amount < pre_sol_amount:
                action = "buy"
                transfer_amount = pre_sol_amount - post_sol_amount
                input_mint = SOL_MINT
                output_mint = non_sol_mint
            elif post_sol_amount > pre_sol_amount:
                action = "sell"
                transfer_amount = post_sol_amount - pre_sol_amount
                input_mint = non_sol_mint
                output_mint = SOL_MINT

        # Check if this is a swap (SOL ↔ Token)
        swap_data = extract_swap_details(inner_instructions, non_sol_mint)

        swap_data_2 = determine_swap_direction(pre_token_balances, post_token_balances)

                # Print the results
        print("Pre Token SOL Balance:", pre_sol_balance)
        print("Post Token SOL Balance:", post_sol_balance)
        print("Post Token non SOL Balance:", pre_non_sol_balance)
        print("Post Token non SOL Balance:", post_non_sol_balance)
        print("direction:", swap_data_2.get("direction"))
        print("Transfer Amount:", swap_data_2.get("sol_amount"))
        print("Non-SOL Mint:", non_sol_mint)
        print("sol_amount:", swap_data_2.get("sol_amount"))
        print("token_amount:", swap_data_2.get("token_amount"))

        # return {
        #     "action": action,
        #     "input_mint": input_mint,
        #     "output_mint": output_mint,
        #     "amount_lamports": transfer_amount
        # }

        if swap_data_2.get("direction") == "sol_to_token":
            action = "buy"
            input_mint = SOL_MINT
            output_mint = non_sol_mint
        elif swap_data_2.get("direction") == "token_to_sol":
            action = "sell"
            input_mint = non_sol_mint
            output_mint = SOL_MINT

        if swap_data:
            return {
                "action": action,
                "input_mint": input_mint,
                "output_mint": output_mint,
                # "swap_direction": swap_data["direction"],
                "amount_lamports": swap_data.get("sol_amount"),
                # "token_amount": swap_data["token_amount"],
                # "token_mint": non_sol_mint,
                # "fee": swap_data.get("fee", 0),
            }

        return {"action": "none"}

    except KeyError as ke:
        return {"error": f"Missing key in transaction: {str(ke)}"}
    except (ValueError, TypeError) as ve:
        return {"error": f"Invalid data type encountered: {str(ve)}"}
    except Exception as e:
        return {"error": f"Unexpected exception occurred: {str(e)}"}

# last_signature = "3goPp61mrt9Ah99WQvP4PJGRRkroziqGQA6Ys6fF3XMDAHySyzDA2NiukTnXENwx6Zj9UgKDAs4tdHEjUy7eFt1Q"
# last_signature = Signature.from_string(last_signature)

def extract_swap_details(inner_instructions, non_sol_mint):
    if not inner_instructions or not non_sol_mint:
        return None

    sol_transfers = []
    token_transfers = []

    for inner in inner_instructions:
        for instr in inner.get("instructions", []):
            if "parsed" not in instr:
                continue

            parsed = instr["parsed"]
            if parsed["type"] == "transferChecked":
                mint = parsed["info"].get("mint")
                amount = float(parsed["info"]["tokenAmount"]["uiAmount"])

                if mint == "So11111111111111111111111111111111111111112":
                    sol_transfers.append(amount)
                elif mint == non_sol_mint:
                    token_transfers.append(amount)

    if not sol_transfers or not token_transfers:
        return None

    # Determine swap direction (SOL → Token or Token → SOL)
    if len(sol_transfers) >= 1 and len(token_transfers) >= 1:
        sol_amount = abs(sol_transfers[0])
        token_amount = abs(token_transfers[0])

        # More robust direction detection
        if sol_amount > 0 and token_amount > 0:
            # A buy is when we receive tokens and send SOL
            is_buy = (sol_transfers[0] < 0 and token_transfers[0] > 0)
            direction = "buy" if is_buy else "sell"
            return {
                "direction": direction,
                "sol_amount": sol_amount,
                "token_amount": token_amount,
            }

    return None

def determine_swap_direction_0(pre_token_balances, post_token_balances):
    # Find the user's SOL and token balances before/after
    sol_balances = {}
    token_balances = {}

    for balance in pre_token_balances + post_token_balances:
        owner = balance.get('owner')
        mint = balance.get('mint')
        amount = float(balance['uiTokenAmount']['uiAmountString'])

        if mint == "So11111111111111111111111111111111111111112":
            sol_balances[owner] = sol_balances.get(owner, {})
            sol_balances[owner]['pre' if balance in pre_token_balances else 'post'] = amount
        else:
            token_balances[owner] = token_balances.get(owner, {})
            token_balances[owner]['pre' if balance in pre_token_balances else 'post'] = amount

    # Find the user's main account (largest SOL balance change)
    user_account = max(
        sol_balances.keys(),
        key=lambda k: abs(sol_balances.get(k, {}).get('pre', 0) - sol_balances.get(k, {}).get('post', 0))
    )

    # Calculate deltas
    sol_delta = sol_balances.get(user_account, {}).get('post', 0) - sol_balances.get(user_account, {}).get('pre', 0)
    token_delta = token_balances.get(user_account, {}).get('post', 0) - token_balances.get(user_account, {}).get('pre', 0)

    # Determine direction
    if sol_delta < 0 and token_delta > 0:
        return "sol_to_token", abs(sol_delta), abs(token_delta)
    elif sol_delta > 0 and token_delta < 0:
        return "token_to_sol", abs(sol_delta), abs(token_delta)
    else:
        return None, 0, 0

def determine_swap_direction(pre_token_balances, post_token_balances):
    print("direction:")
    try:
        # Input validation
        if not pre_token_balances or not post_token_balances:
            print("Empty token balance lists provided")
            print(f"pre_token_balances: {pre_token_balances}")
            print(f"post_token_balances: {post_token_balances}")
            return None

        user_accounts = set()
        balance_changes = {}

        # Process pre and post balances
        for balance_type, balances in [('pre', pre_token_balances), ('post', post_token_balances)]:
            try:
                for balance in balances:
                    try:
                        owner = balance['owner']
                        mint = balance['mint']
                        amount_str = balance['uiTokenAmount']['uiAmountString']
                        amount = float(amount_str)

                        if owner not in balance_changes:
                            balance_changes[owner] = {
                                'sol': {'pre': 0, 'post': 0},
                                'token': {'pre': 0, 'post': 0}
                            }

                        key = balance_type  # 'pre' or 'post'

                        if mint == "So11111111111111111111111111111111111111112":
                            balance_changes[owner]['sol'][key] = amount
                        else:
                            balance_changes[owner]['token'][key] = amount

                        user_accounts.add(owner)

                        # Debug print for each processed balance
                        print(f"Processed {balance_type} balance - Owner: {owner}, Mint: {mint}, Amount: {amount}")

                    except KeyError as e:
                        print(f"Missing expected key in balance data: {e}")
                        print(f"Problematic balance entry: {balance}")
                        continue
                    except ValueError as e:
                        print(f"Failed to parse amount for {owner}: {amount_str}")
                        continue

            except TypeError as e:
                print(f"Invalid balance list format: {e}")
                print(f"Balance list that failed: {balances}")
                return None

        # Print all collected balance changes
        print("\nAll balance changes collected:")
        for owner, changes in balance_changes.items():
            print(f"Account: {owner}")
            print(f"  SOL - pre: {changes['sol']['pre']}, post: {changes['sol']['post']}")
            print(f"  Token - pre: {changes['token']['pre']}, post: {changes['token']['post']}")
            print("---")

        if not user_accounts:
            print("No valid user accounts found in balance changes")
            return None

        # Find main account with largest SOL movement
        try:
            main_account = max(
                user_accounts,
                key=lambda acc: abs(balance_changes[acc]['sol']['post'] - balance_changes[acc]['sol']['pre'])
            )
            print(f"\nMain account identified: {main_account}")
        except KeyError as e:
            print(f"Missing SOL balance data for accounts: {e}")
            print("Available accounts and their SOL balances:")
            for acc in user_accounts:
                print(f"{acc}: {balance_changes[acc]['sol']}")
            return None

        # Calculate deltas
        try:
            sol_delta = balance_changes[main_account]['sol']['post'] - balance_changes[main_account]['sol']['pre']
            token_delta = balance_changes[main_account]['token']['post'] - balance_changes[main_account]['token']['pre']
            print(f"\nDeltas for main account {main_account}:")
            print(f"SOL delta: {sol_delta}")
            print(f"Token delta: {token_delta}")
        except KeyError as e:
            print(f"Missing balance data for {main_account}: {e}")
            print(f"Available data for this account: {balance_changes[main_account]}")
            return None

        # Determine swap direction
        try:
            if sol_delta < 0 and token_delta > 0:  # SOL spent, tokens received
                output_mint = next(
                    b['mint'] for b in post_token_balances
                    if b['mint'] != "So11111111111111111111111111111111111111112"
                )
                result = {
                    'direction': 'sol_to_token',
                    'action': 'buy',
                    'sol_amount': abs(sol_delta),
                    'token_amount': token_delta,
                    'input_mint': 'So11111111111111111111111111111111111111112',
                    'output_mint': output_mint
                }
                print("\nDetermined swap direction:")
                print(f"Type: SOL to Token (Buy)")
                print(f"SOL spent: {abs(sol_delta)}")
                print(f"Tokens received: {token_delta}")
                print(f"Token mint: {output_mint}")
                return result

            elif sol_delta > 0 and token_delta < 0:  # Tokens spent, SOL received
                input_mint = next(
                    b['mint'] for b in pre_token_balances
                    if b['mint'] != "So11111111111111111111111111111111111111112"
                )
                result = {
                    'direction': 'token_to_sol',
                    'action': 'sell',
                    'sol_amount': sol_delta,
                    'token_amount': abs(token_delta),
                    'input_mint': input_mint,
                    'output_mint': 'So11111111111111111111111111111111111111112'
                }
                print("\nDetermined swap direction:")
                print(f"Type: Token to SOL (Sell)")
                print(f"Tokens spent: {abs(token_delta)}")
                print(f"SOL received: {sol_delta}")
                print(f"Token mint: {input_mint}")
                return result

            else:
                print("\nNo clear swap direction detected")
                print("Possible reasons:")
                print("- Both SOL and token amounts increased (possible liquidity deposit)")
                print("- Both SOL and token amounts decreased (possible liquidity withdrawal)")
                print("- No significant balance changes detected")
                return None

        except StopIteration:
            print("\nCould not find non-SOL mint in token balances")
            print("Available mints in post_token_balances:")
            for b in post_token_balances:
                print(f"- {b['mint']}")
            return None
        except Exception as e:
            print(f"\nError determining swap direction: {e}")
            print("Current state at time of failure:")
            print(f"Main account: {main_account}")
            print(f"SOL delta: {sol_delta}")
            print(f"Token delta: {token_delta}")
            return None

    except Exception as e:
        print(f"\nUnexpected error in determine_swap_direction: {e}")
        return None


# Step 3: Monitor transactions for a wallet address
def monitor_wallet(wallet_address):
    last_signature = None  # Track the last seen transaction signature

    try:
        # Convert the wallet address string to a Pubkey object
        wallet_pubkey = Pubkey.from_string(wallet_address)
    except Exception as e:
        print(f"Error converting wallet address to Pubkey: {e}")
        return

    while True:
        try:
            try:
                # Fetch recent transactions for the wallet address
                signatures_response = solana_client.get_signatures_for_address(wallet_pubkey, limit=20)
            except Exception as e:
                print(f"Error fetching transaction signatures: {e}")
                time.sleep(5)
                continue  # Skip this iteration and retry

            if signatures_response and signatures_response.value:
                signatures = signatures_response.value  # List of recent transaction signatures

                if last_signature is None:
                    print("First-time detection: Searching for a valid transaction...")

                    for sig in signatures:
                        try:
                            latest_signature = sig.signature

                            # last_signature = "4hpMVSEFroDC9296vLrDuCsGydFvAFesER9a1SLkNoLK7hCQ5vCoX9Dv32scS7jE1ymLHzvUPQYBT85FEvW57VPJ"
                            last_signature = "SK38kAh8o1nQi11hvssnDxJ8W9rcvDoVGaH5NpY4MLgLuLYDhSJ41HAcPn51KyXaUqBtPHod3mkjE9H4RuXhanH"
                            latest_signature = Signature.from_string(last_signature)

                            # Wait before analyzing the new transaction
                            time.sleep(5)

                            # Fetch the transaction details
                            transaction_response = solana_client.get_transaction(
                                latest_signature,
                                commitment="finalized",
                                encoding="jsonParsed",
                                max_supported_transaction_version=0
                            )

                            if transaction_response and transaction_response.value:
                                transaction = transaction_response.value
                                print(transaction)
                                print(f"Analyzing transaction: {latest_signature}")
                                parsed_data = analyze_transaction(transaction)

                                print(parsed_data)

                                # Check if transaction meets the required conditions
                                if (parsed_data.get("input_mint") != "So11111111111111111111111111111111111111112" or
                                    parsed_data.get("output_mint") != "So11111111111111111111111111111111111111112") and \
                                        parsed_data.get("amount_lamports", 0) > 0:

                                    print(f"Valid transaction found: {latest_signature}")
                                    # //////////////////////////////////////////////////// replace with db
                                    last_signature = latest_signature
                                    # break  # Stop searching once a valid transaction is found

                        except Exception as e:
                            print(f"Error analyzing transaction {sig.signature}: {e}")
                            continue  # Continue checking other transactions

                    if last_signature is None:
                        print("No valid transaction found in the latest signatures.")

                else:
                    # Normal case: Process all new signatures until last_signature is found
                    new_signatures = []
                    for sig in signatures:
                        try:
                            if sig.signature == last_signature:
                                break  # Stop when we reach the last known signature
                            new_signatures.append(sig.signature)
                        except Exception as e:
                            print(f"Error processing signature: {e}")
                            continue  # Continue checking other transactions

                    # If new signatures exist, analyze them
                    if new_signatures:
                        print(f"New transactions detected: {len(new_signatures)}")

                        # Process transactions in chronological order (oldest first)
                        for sig in reversed(new_signatures):
                            try:
                                print(f"Waiting 5 seconds before analyzing transaction: {sig}")
                                time.sleep(5)  # Wait before analyzing

                                # Fetch the transaction details
                                transaction_response = solana_client.get_transaction(
                                    sig,
                                    commitment="finalized",
                                    encoding="jsonParsed",
                                    max_supported_transaction_version=0
                                )

                                if transaction_response and transaction_response.value:
                                    transaction = transaction_response.value
                                    parsed_data = analyze_transaction(transaction)
                                    print(parsed_data)
                                else:
                                    print(f"Failed to fetch transaction details for {sig}")

                            except Exception as e:
                                print(f"Error analyzing transaction {sig}: {e}")
                                continue  # Continue with the next transaction

                        # Update last_signature to the newest processed signature
                        last_signature = new_signatures[0]

        except Exception as e:
            print(f"Unexpected error in monitoring loop: {e}")

        time.sleep(5)  # Poll every 5 seconds

# Replace with the wallet address you want to monitor
wallet_address = "3KNCdquQuPBq6ZWChRJr8jGpkoyZ5LurLCt6sNJJMxbq"
monitor_wallet(wallet_address)

