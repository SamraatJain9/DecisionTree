def generate_4ps_questions(is_product):
    # Questions for Product
    product_questions = {
        "Product": [
            "What product or service do we offer?",
            "What are the key features and differentiators?",
            "Are there different variations or models available?",
            "What is the product's lifecycle?",
            "What is the product's warranty or guarantee?",
            "How do we ensure product quality and consistency?",
            "What is the product's packaging like?",
            "What after-sales support is provided?",
            "How does the product align with current market trends?",
            "What is the product's environmental impact?"
        ],
        "Pricing": [
            "What is our pricing strategy?",
            "Are there different pricing tiers or packages?",
            "Is pricing fixed or customized per customer?",
            "Do we offer discounts, rebates, or promotions?",
            "What are the payment terms and methods?",
            "How does our pricing compare to competitors?",
            "What is the perceived value of our product or service?",
            "How do we handle refunds and returns?",
            "What is the impact of our pricing on brand perception?",
            "How do we monitor and adjust pricing over time?"
        ],
        "Promotion": [
            "What are our primary communication channels?",
            "Who is responsible for communication efforts?",
            "How frequently do we engage with potential customers?",
            "What is our content strategy?",
            "What is our messaging strategy?",
            "What is our tone of voice?",
            "How do we handle public relations and media outreach?",
            "What partnerships or collaborations can enhance our promotional efforts?",
            "How do we measure the effectiveness of our promotional activities?",
            "What is our crisis communication plan?"
        ],
        "Placement": [
            "Where do we sell our product or service?",
            "Do we focus on domestic and/or international markets?",
            "Do we have resellers or other partners involved in the sales process?",
            "Where is our product manufactured and stored?",
            "What distribution channels do we use?",
            "How do we ensure product availability and stock levels?",
            "What is our delivery and fulfillment strategy?",
            "How do we handle returns and exchanges?",
            "What is our strategy for entering new markets?",
            "How do we manage relationships with distributors and retailers?"
        ]
    }

    # Questions for Service
    service_questions = {
        "Product": [
            "What service do we provide?",
            "Are there different types of service offerings?",
            "What are the key features and/or differentiators?",
            "How do we ensure service quality?",
            "What is the customer experience like?",
            "How do we support and maintain the service over time?",
            "What after-service or ongoing support is offered?"
        ],
        "Pricing": [
            "What is our pricing?",
            "Do we adhere to a specific pricing strategy?",
            "Are there different pricing tiers for different service levels?",
            "Is pricing fixed, or do we create a customer price for each customer?",
            "Do we offer discounts or rebates?",
            "Which payment methods do we accept and which do we prefer?",
            "What are the pricing terms?",
            "Do we offer refunds?",
            "How do we compare with competitors’ pricing?"
        ],
        "Promotion": [
            "How do we communicate with potential customers?",
            "Who communicates with them?",
            "What channels do we use?",
            "How frequently do we communicate with potential customers?",
            "How do we frame our service and its benefits?",
            "What is our content strategy?",
            "What is our messaging strategy?",
            "What is our tone of voice?"
        ],
        "Placement": [
            "Where do we provide our service?",
            "Do we focus on domestic and/or international markets?",
            "Do we have resellers or other partners involved in delivering the service?",
            "Where is the service delivered from?",
            "What channels do we use for service distribution?"
        ]
    }

    # Based on whether it is a product or service, choose appropriate questions
    if is_product:
        questions = product_questions
    else:
        questions = service_questions

    return questions


def create_4ps_document(is_product):
    # Get relevant questions for product or service
    questions = generate_4ps_questions(is_product)

    # Create file name based on whether it's a product or service
    file_name = "4Ps_Questions_Product.txt" if is_product else "4Ps_Questions_Service.txt"

    # Open the file for writing
    with open(file_name, 'w') as file:
        for section, question_list in questions.items():
            file.write(f"### {section} ###\n")
            for question in question_list:
                file.write(f"{question}\n")
                file.write("Answer: _________________________\n\n")
    
    print(f"Questions document '{file_name}' has been created!")


def main():
    # Ask the user if they are dealing with a product or service
    answer = input("Is it a product or a service? (Type 'product' or 'service'): ").strip().lower()

    if answer == 'product':
        create_4ps_document(is_product=True)
    elif answer == 'service':
        create_4ps_document(is_product=False)
    else:
        print("Invalid input. Please type 'product' or 'service'.")


if __name__ == "__main__":
    main()
