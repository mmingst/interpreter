import re
import sys
import os
from enum import Enum
from LoopsConditionalsFunctions import *

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis

INTEGER       = 'INTEGER'
BOOLEAN       = 'BOOLEAN'
ARRAY         = 'ARRAY'
PLUS          = 'PLUS'
MINUS         = 'MINUS'
MUL           = 'MUL'
NOT           = 'NOT'
AND           = 'AND'
OR            = 'OR'
EQUALS        = 'EQUALS'
LESS          = 'LESS'
LPAREN        = 'LPAREN'
RPAREN        = 'RPAREN'
LBRACK        = 'LBRACK'
RBRACK        = 'RBRACK'
ID            = 'ID'
ASSIGN        = 'ASSIGN'
LCBRACK       = 'LCBRACK'
RCBRACK       = 'RCBRACK'
SEMI          = 'SEMI'
DOT           = 'DOT'
FUNCTION      = 'FUNCTION'
IF            = 'IF'
ELSE          = 'ELSE'
WHILE         = 'WHILE'
RETURN        = 'RETURN'
ACCESS        = 'ACCESS'
LEN           = 'LEN'
VAR           = 'VAR'
COLON         = 'COLON'
COMMA         = 'COMMA'
EOF           = 'EOF'


class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __str__(self):
        """String representation of the class instance.

        Examples:
            Token(INTEGER, 3)
            Token(PLUS, '+')
            Token(MUL, '*')
        """
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()


RESERVED_KEYWORDS = {
    'FUNCTION': Token('FUNCTION', 'FUNCTION'),
    'if': Token('IF', 'If'),
    'else': Token('ELSE', 'Else'),    
    'while': Token('WHILE', 'while'),
    'return': Token('RETURN', 'return'),
    'len' : Token('LEN', 'len'),
    'ACCESS': Token('ACCESS', 'ACCESS'),
    'VAR': Token('VAR', 'VAR'),
    'INTEGER': Token('INTEGER', 'INTEGER'),
    'BOOLEAN' : Token('BOOLEAN', 'BOOLEAN'),
    'ARRAY': Token('ARRAY', 'ARRAY'),
    'True' : Token('BOOLEAN', 'True'),
    'False' : Token('BOOLEAN', 'False'),  
}

TYPE_DICT = {}

PARAM_DICT = {}

class Lexer(object):
    def __init__(self, text):
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]

    def peek(self):
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def number(self):
        """Return a (multidigit) integer consumed from the input."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()

        token = Token('INTEGER', int(result))

        return token
    

    def _id(self):
        """Handle identifiers and reserved keywords"""
        result = ''
        while self.current_char is not None and self.current_char.isalnum():
            result += self.current_char
            self.advance()

        token = RESERVED_KEYWORDS.get(result, Token(ID, result))
        return token

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char.isalpha():
                return self._id()

            if self.current_char.isdigit():
                return self.number()

            if self.current_char == ':' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(ASSIGN, ':=')
            
            if self.current_char == '=' and self.peek() == '=':
                self.advance()
                self.advance()
                return Token(EQUALS, '==')
            
            if self.current_char == '<':
                self.advance()
                return Token(LESS, '<')

            if self.current_char == ';':
                self.advance()
                return Token(SEMI, ';')

            if self.current_char == ':':
                self.advance()
                return Token(COLON, ':')

            if self.current_char == ',':
                self.advance()
                return Token(COMMA, ',')

            if self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')

            if self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')

            if self.current_char == '*':
                self.advance()
                return Token(MUL, '*')

            if self.current_char == '&' and self.peek() == '&':
                self.advance()
                self.advance()
                return Token(AND, '&&')

            if self.current_char == '|' and self.peek() == '|':
                self.advance()
                self.advance()
                return Token(OR, '||')

            if self.current_char == '!':
                self.advance()
                return Token(NOT, '!')

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')
            
            if self.current_char == '{':
                self.advance()
                return Token(LCBRACK, '{')

            if self.current_char == '}':
                self.advance()
                return Token(RCBRACK, '}')

            if self.current_char == '[':
                self.advance()
                return Token(LBRACK, '[')
            
            if self.current_char == ']':
                self.advance()
                return Token(RBRACK, ']')

            if self.current_char == '.':
                self.advance()
                return Token(DOT, '.')

            self.error()

        return Token(EOF, None)


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################  
class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class Bool(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class Array(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class UnaryOp(AST):
    def __init__(self, op, expr):
        self.token = self.op = op
        self.expr = expr


class Compound(AST):
    """Represents a 'BEGIN ... END' block"""
    def __init__(self):
        self.children = []



class Assignment(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right


class Var(AST):
    """The Var node is constructed out of ID token."""
    def __init__(self, token):
        self.token = token
        self.value = token.value


class NoOp(AST):
    pass


class Function(AST):
    def __init__(self, name, params, block):
        self.name = name
        self.params = params
        self.block = block

class Param(AST):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node

class IfCond(AST):
    def __init__(self, cond, exp1):
        self.cond = cond
        self.exp1 = exp1

class IfElseCond(AST):
    def __init__(self, cond, exp1, exp2):
        self.cond = cond
        self.exp1 = exp1
        self.exp2 = exp2

class WhileCond(AST):
    def __init__(self, cond, exp):
        self.cond = cond
        self.exp = exp

class Len(AST):
    def __init__(self, name):
        self.name = name

class ReturnExpr(AST):
    def __init__(self, expr):
        self.expr = expr

class ArrAccess(AST):
    def __init__(self, name, index):
        self.name = name
        self.index = index

class Block(AST):
    def __init__(self, declarations, compound_statement):
        self.declarations = declarations
        self.compound_statement = compound_statement

class VarDecl(AST):
    def __init__(self, var_node, type_node):
        self.var_node = var_node
        self.type_node = type_node

class Type(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            print("Error eating: " + str(self.current_token))
            self.error()

    def function(self):
        """function : (FUNCTION ID (LPAREN formal_parameter_list RPAREN)? COLON block SEMI)*"""
        self.eat(FUNCTION)
        var_node = self.variable()
        function_name = var_node.value
        
        params = []
        
        self.eat(LPAREN)
        if self.current_token.type != RPAREN:
            params = self.formal_parameter_list()

        self.eat(RPAREN)
        self.eat(COLON)
        
        block_node = self.block()
        function_node = Function(function_name, params, block_node)
        self.eat(DOT)
        return function_node

    def formal_parameter_list(self):
        """ formal_parameter_list : formal_parameters
                                | formal_parameters SEMI formal_parameter_list
        """
        
        if not self.current_token.type == ID:
            return []

        param_nodes = self.formal_parameters()

        while self.current_token.type == SEMI:
            self.eat(SEMI)
            param_nodes.extend(self.formal_parameters())

        return param_nodes

    def formal_parameters(self):
        """ formal_parameters : ID (COMMA ID)* COLON type_spec """
        param_nodes = []

        param_tokens = [self.current_token]
        self.eat(ID)
        while self.current_token.type == COMMA:
            self.eat(COMMA)
            param_tokens.append(self.current_token)
            self.eat(ID)

        self.eat(COLON)
        type_node = self.type_spec()
        
        for param_token in param_tokens:
            TYPE_DICT[param_token.value] = type_node.value
            PARAM_DICT[param_token.value] = type_node.value
            param_node = Param(Var(param_token), type_node)
            param_nodes.append(param_node)

        return param_nodes
    
    def variable_declaration(self):
        """variable_declaration : ID (COMMA ID)* COLON type_spec"""
        var_nodes = [Var(self.current_token)]  # first ID
        self.eat(ID)

        while self.current_token.type == COMMA:
            self.eat(COMMA)
            var_nodes.append(Var(self.current_token))
            self.eat(ID)

        self.eat(COLON)

        type_node = self.type_spec()
        for var_node in var_nodes:
            TYPE_DICT[var_node.value] = type_node.value

        var_declarations = [
            VarDecl(var_node, type_node)
            for var_node in var_nodes
        ]
        
        return var_declarations

    def type_spec(self):
        """type_spec : INTEGER
                     | BOOLEAN
                     | ARRAY
        """
        token = self.current_token
        if self.current_token.type == INTEGER:
            self.eat(INTEGER)
        elif self.current_token.type == BOOLEAN:
            self.eat(BOOLEAN)
        elif self.current_token.type == ARRAY:
            self.eat(ARRAY)
        node = Type(token)
        return node

    def block(self):
        """block : declarations compound_statement"""
        declaration_nodes = self.declarations()
        compound_statement_node = self.compound_statement()
        node = Block(declaration_nodes, compound_statement_node)
        return node

    def declarations(self):
        """declarations : (VAR (variable_declaration SEMI)+)*
                    | empty
        """
        declarations = []
        if self.current_token.type == VAR:
            self.eat(VAR)
            while self.current_token.type == ID:
                var_decl = self.variable_declaration()
                declarations.extend(var_decl)
                self.eat(SEMI)

        return declarations

    def compound_statement(self):
        """
        compound_statement: LCBRACK statement_list RCBRACK
        """
        self.eat(LCBRACK)
        nodes = self.statement_list()
        self.eat(RCBRACK)

        root = Compound()
        for node in nodes:
            root.children.append(node)

        return root

    def statement_list(self):
        """
        statement_list : statement
                       | statement SEMI statement_list
        """
        node = self.statement()

        results = [node]

        while self.current_token.type == SEMI:
            self.eat(SEMI)
            results.append(self.statement())

        return results

    def statement(self):
        """
        statement : compound_statement
                  | if_statement
                  | while_statement
                  | return_statement
                  | assignment_statement
                  | empty
        """
        if self.current_token.type == LCBRACK:
            node = self.compound_statement()
        elif self.current_token.type == RETURN:
            node = self.return_statement()
        elif self.current_token.type == IF:
            node = self.if_statement()
        elif self.current_token.type == WHILE:
            node = self.while_statement()
        elif self.current_token.type == ID:
            node = self.assignment_statement()
        else:
            node = self.empty()
        return node

    def if_statement(self):
        """
        if_statement : IF expr COLON compound_statement (ELSE COLON compound_statement)
        """
        self.eat(IF)
        cond = self.expr()
        exp1 = self.compound_statement()

        if self.current_token.type == ELSE:
            self.eat(ELSE)
            exp2 = self.compound_statement()
            node = IfElseCond(cond, exp1, exp2)
        else:
            node = IfCond(cond, exp1)
        return node

    def while_statement(self):
        """
        while_statement : WHILE expr COLON compound_statement
        """
        self.eat(WHILE)
        cond = self.expr()
        exp = self.compound_statement()        
        node = WhileCond(cond, exp)
        return node

    def assignment_statement(self):
        """
        assignment_statement : variable ASSIGN expr
        """
        left = self.variable()
        token = self.current_token
        self.eat(ASSIGN)
        if TYPE_DICT[left.value] in ('INTEGER', 'BOOLEAN', 'ARRAY'):
            right = self.expr()
        node = Assignment(left, token, right)
        return node

    def return_statement(self):
        """
        return_statement : RETURN expr
        """
        self.eat(RETURN)
        expr = self.expr()
        node = ReturnExpr(expr)
        return node
    
    def length_check(self):
        """
        length_check : LEN LPAR expr RPAR
        """
        self.eat(LEN)
        self.eat(LPAREN)
        name = self.variable()
        node = Len(name)
        self.eat(RPAREN)
        return node

    def variable(self):
        """
        variable : ID
        """
        name = Var(self.current_token)
        self.eat(ID)
        if self.current_token.type == LBRACK:
            self.eat(LBRACK)
            index = self.expr()
            node = ArrAccess(name, index)
            self.eat(RBRACK)
        else:
            node = name
        return node

    def empty(self):
        """An empty production"""
        return NoOp()

    def expr(self):
        """
        expr : term ((PLUS | MINUS | AND | OR | NOT | EQUALS | LESS) term)*
        """
        node = self.term()

        while self.current_token.type in (PLUS, MINUS, AND, OR, NOT, EQUALS, LESS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
            elif token.type == MINUS:
                self.eat(MINUS)
            elif token.type == AND:
                self.eat(AND)
            elif token.type == OR:
                self.eat(OR)
            elif token.type == NOT:
                self.eat(NOT)
            elif token.type == EQUALS:
                self.eat(EQUALS)
            elif token.type == LESS:
                self.eat(LESS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def term(self):
        """term : factor ((MUL factor)*"""
        node = self.factor()
        
        while self.current_token.type in (MUL):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
        
            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def factor(self):
        """factor : PLUS factor
                  | MINUS factor
                  | NOT factor
                  | INTEGER_CONST
                  | REAL_CONST
                  | LPAREN expr RPAREN
                  | LEN LPAREN variable RPAREN
                  | variable
        """

        token = self.current_token
        if token.type == PLUS:
            self.eat(PLUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == MINUS:
            self.eat(MINUS)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == NOT:
            self.eat(NOT)
            node = UnaryOp(token, self.factor())
            return node
        elif token.type == INTEGER:
            self.eat(INTEGER)
            return Num(token)
        elif token.type == BOOLEAN:
            self.eat(BOOLEAN)
            return Bool(token)
        elif token.type == ARRAY:
            self.eat(ARRAY)
            return Array(token)
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node
        elif token.type == LEN:
            node = self.length_check()
            return node
        else:
            node = self.variable()
            return node

    def parse(self):
        node = self.function()
        if self.current_token.type != EOF:
            self.error()

        return node


###############################################################################
#                                                                             #
#  AST visitors (walkers)                                                     #
#                                                                             #
###############################################################################

class NodeVisitor(object):
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))



###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################

class Interpreter(NodeVisitor):
    def __init__(self, tree):
        self.tree = tree

    def visit_Function(self, node):
        print("Visiting function")
        function_name = node.name
        function_params = []
        for param_name, param_type in PARAM_DICT.items():
            if param_type == 'INTEGER':
                function_params.append(IntVar(param_name))
            elif param_type == 'BOOLEAN':
                function_params.append(BoolVar(param_name))
            elif param_type == 'ARRAY':
                function_params.append(ArrayVar(param_name))
        function_body = self.visit(node.block)
        return FunctionDefine(function_name, function_params, Do(function_body))

    def visit_Block(self, node):
        for declaration in node.declarations:
            self.visit(declaration)
        return self.visit(node.compound_statement)

    def visit_VarDecl(self, node):
        print("Visiting vardecl")
        pass

    def visit_Type(self, node):
        print("Visiting type")
        # Do nothing
        pass

    def visit_BinOp(self, node):
        print("Visiting binop")
        lval = self.visit(node.left)
        rval = self.visit(node.right)
        if node.op.type == PLUS:
            return Plus(lval, rval)
        elif node.op.type == MUL:
            return Times(lval, rval)
        elif node.op.type == MINUS:
            return Plus(lval, Times(IntVal(-1), rval))
        elif node.op.type == AND:
            return And(lval, rval)
        elif node.op.type == OR:
            return Or(lval, rval)
        elif node.op.type == EQUALS:
            return Equals(lval, rval)
        elif node.op.type == LESS:
            return Less(lval, rval)
        else:
            sys.exit("Op not supported: (We can't do division yet")

    def visit_Num(self, node):
        print("Visiting number")
        return IntVal(node.value)

    def visit_Bool(self, node):
        print("Visiting bool")
        return BoolVal(node.value)
        
    def visit_UnaryOp(self, node):
        print("Visiting unop")
        if node.op.type == PLUS:
            return Times(IntVal(1), self.visit(node.expr))
        elif node.op.type == MINUS:
            return Times(IntVal(-1), self.visit(node.expr))
        elif node.op.type == NOT:
            return Not(self.visit(node.expr))

    def visit_Compound(self, node):
        print("Visiting compound")
        results = []
        for child in node.children:
            result = self.visit(child)
            if result is None:
                continue
            results.append(result)
        return results

    def visit_IfCond(self, node):
        print("Visiting if")
        cond = self.visit(node.cond)
        exp1 = self.visit(node.exp1)
        return If(cond, Do(exp1))
    
    def visit_IfElseCond(self, node):
        print("Visiting if")
        cond = self.visit(node.cond)
        exp1 = self.visit(node.exp1)
        exp2 = self.visit(node.exp2)
        return IfElse(cond, Do(exp1), Do(exp2))

    def visit_WhileCond(self, node):
        print("Visiting while")
        cond = self.visit(node.cond)
        exp = self.visit(node.exp)
        return While(cond, Do(exp))

    def visit_ReturnExpr(self, node):
        print("Visiting return")
        return_object = self.visit(node.expr)
        return Return(return_object)

    def visit_ArrAccess(self, node):
        print("Visiting access")
        array_name = self.visit(node.name)
        array_index = self.visit(node.index)
        return Access(array_name, array_index)

    def visit_Assignment(self, node):
        print("Visiting assign")
        var_name = node.left.value
        if TYPE_DICT[var_name] == 'INTEGER':
            var_object = IntVar(var_name)
        elif TYPE_DICT[var_name] == 'BOOLEAN':
            var_object = BoolVar(var_name)
        elif TYPE_DICT[var_name] == 'ARRAY':
            var_object = ArrayVar(var_name)
        var_value = self.visit(node.right)
        return Assign(var_object, var_value)

    def visit_Len(self, node):
        print("Visiting length")
        var_name = self.visit(node.name)
        return Length(var_name)
        
    def visit_Var(self, node):
        print("Visiting var")
        var_name = node.value
        if TYPE_DICT[var_name] == 'INTEGER':
            var_object = IntVar(var_name)
        elif TYPE_DICT[var_name] == 'BOOLEAN':
            var_object = BoolVar(var_name)
        elif TYPE_DICT[var_name] == 'ARRAY':
            var_object = ArrayVar(var_name)
        return var_object

    def visit_NoOp(self, node):
        print("Visiting noOp")
        pass

    def interpret(self):
        tree = self.tree
        if tree is None:
            return ''
        return self.visit(tree)


def main():
    import sys
    rfile = input("Convert File: ")
    text = open(rfile, 'r').read()

    lexer = Lexer(text)
    parser = Parser(lexer)
    tree = parser.parse()
    
    # print('\n' + "PARAMETERS:")
    # print('\n' + str(PARAM_DICT) + '\n')

    # print('\n' + "DECLARATIONS:")
    # print('\n' + str(TYPE_DICT) + '\n')
    interpreter = Interpreter(tree)
    interpretation = interpreter.interpret()
    
    inp = input("Print object (y/n)? >>: ")
    if inp == "y":
        print('\n' + "GUISE CONTENTS:")
        print('\n' + str(interpretation) + ".eval(env)" + '\n')
    
    # env = {}
    # print(env)

    # store = {}
    # solver = z3.Solver()
    # x = Do([interpretation, 
    # Assign(ArrayVar('q'), ArraySym(IntVal(2), 'Q')), 
    # Assign(ArrayVar('s'), ArraySym(IntVal(1), 'S')), 
    # AssignFunc(IntVar('x'), 
    # FunctionCall('TestHighMiddleLow', [ArrayVar('q'), ArrayVar('s')]))]).leafsymbex(store, solver)

    # print(x)


if __name__ == '__main__':
    main()
