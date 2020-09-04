function val = loss_gmf(A, B, U, V)
val = norm(U - A \ (B * V), 'fro')^2 ;
end