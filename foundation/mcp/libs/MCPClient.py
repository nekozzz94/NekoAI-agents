from typing import Optional
from contextlib import AsyncExitStack 
from mcp import ClientSession
from mcp.client.sse import sse_client

class MCPClient:
    """A client for interacting with a MCP (Model Context Protocol) server.    
    Attributes:
        session (Optional[ClientSession]): The active client session with the MCP server.
        exit_stack (AsyncExitStack): Context manager for handling async resources.
    """
    def __init__(self, mcp_server_url):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.mcp_server_url = mcp_server_url
        
    async def __aenter__(self):
        """Establishes an async connection to the MCP server using SSE transport.
        
        Args:
            mcp_server_url (str): The URL endpoint of the MCP server to connect to.
            
        Returns:
            ClientSession: The established client session object.
            
        Raises:
            ConnectionError: If the connection to the server cannot be established.
        """

        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(self.mcp_server_url)
        )
        read, write = sse_transport

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        await self.session.initialize()       
                
        return self
    
    async def get_tools(self):
        """Make a call to MCP server
        
        Returns:
            Response from MCP server
                
        Raises:
            RuntimeError: If called before establishing a server connection.
        """
        response = await self.session.list_tools()
        tool_names = [tool.name for tool in response.tools]
        print(f'Available Server Tools: {tool_names}')
        
        return response
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanly disconnects from the MCP server.
        
        Closes the async exit stack and cleans up the client session.
        After disconnection, the client will need to reconnect before making
        further server requests.
        """
        await self.exit_stack.aclose()
        self.session = None
