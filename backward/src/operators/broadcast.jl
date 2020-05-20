"""
    ComputGraphStyle <: Broadcast.BroadcastStyle

This style of broadcast will forward the broadcast expression
to be registered in a computation graph, rather than directly
calculate it.
"""
struct ComputGraphStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:AbstractNode}) = ComputGraphStyle()
Broadcast.BroadcastStyle(s::ComputGraphStyle, x::Broadcast.BroadcastStyle) = s

# # this enables method traits broadcast as a constant
Broadcast.broadcastable(x::AbstractNode) = x

function Broadcast.broadcasted(::ComputGraphStyle, f, args...)
    mt = Trait.Broadcasted(f)
    register(mt, args...)
end

Broadcast.materialize(x::AbstractNode) = register(Broadcast.materialize, x)

# NOTE: we don't need it anymore since we implemented backward for materialize
# gradient(::typeof(Broadcast.materialize), grad, output, x) = (grad, )

# directly forward to broadcasted
function backward(node::CachedNode, ::typeof(Broadcast.materialize), grad, out_size)
    # backward_type_assert(node, grad)
    # TODO: replace with @assert when there is a compiler option for it
    # @boundscheck backward_size_assert(node, grad)
    backward(arg(node, 1), grad, out_size)
end

#
function backward(node::CachedNode, ::Trait.Broadcasted, grad, out_size)
    grad_inputs = gradient(node, grad, out_size)
    for (each, each_grad) in zip(args(node), grad_inputs)
        backward(each, each_grad, out_size)
    end
    nothing
end

# arraymath.jl
for sym in (:(/), :(\), :*, :+, :-)
    f = Expr(:., :Base, QuoteNode(sym))

    if f != :/
        @eval ($f)(A::Number, B::Variable{<:AbstractArray}) = broadcast($f, A, B)
        @eval ($f)(A::Number, B::CachedNode{<:Node, <:AbstractArray}) = broadcast($f, A, B)
    end
    if f != :\
        @eval ($f)(A::Variable{<:AbstractArray}, B::Number) = broadcast($f, A, B)
        @eval ($f)(A::CachedNode{<:Node, <:AbstractArray}, B::Number) = broadcast($f, A, B)
    end
end

for sym in (:-, :conj, :real, :imag)
    f = Expr(:., :Base, QuoteNode(sym))
    @eval ($f)(A::Variable{<:AbstractArray}) = broadcast($f, A)
    @eval ($f)(A::CachedNode{<:Node, <:AbstractArray}) = broadcast($f, A)
end
