package com.subbulakshmi.deepfake.config;

import org.hibernate.dialect.PostgreSQLDialect;
import org.hibernate.dialect.PostgreSQLSqlAstTranslator;
import org.hibernate.dialect.pagination.LimitHandler;
import org.hibernate.dialect.pagination.LimitOffsetLimitHandler;
import org.hibernate.engine.spi.SessionFactoryImplementor;
import org.hibernate.sql.ast.SqlAstTranslator;
import org.hibernate.sql.ast.SqlAstTranslatorFactory;
import org.hibernate.sql.ast.spi.StandardSqlAstTranslatorFactory;
import org.hibernate.sql.ast.tree.Statement;
import org.hibernate.sql.ast.tree.select.QueryPart;
import org.hibernate.sql.exec.spi.JdbcOperation;

/**
 * {@link PostgreSQLDialect} for servers older than PostgreSQL 12.
 * <p>
 * Hibernate 6 paginates with {@code OFFSET ? ROWS FETCH FIRST ? ROWS ONLY}, but PostgreSQL releases before 11 only
 * accept a constant or a parenthesised expression after {@code FETCH FIRST}, so a bound parameter is a syntax error.
 * HQL/JPQL paging is rendered by the dialect's SQL AST translator (the {@link LimitHandler} only covers native
 * queries), so both are switched here to {@code LIMIT ? OFFSET ?}, which every PostgreSQL version understands.
 * Activate it with the {@code legacypg} Spring profile.
 */
public class LegacyPostgresDialect extends PostgreSQLDialect {

    @Override
    public LimitHandler getLimitHandler() {
        return LimitOffsetLimitHandler.INSTANCE;
    }

    @Override
    public SqlAstTranslatorFactory getSqlAstTranslatorFactory() {
        return new StandardSqlAstTranslatorFactory() {
            @Override
            protected <T extends JdbcOperation> SqlAstTranslator<T> buildTranslator(
                    SessionFactoryImplementor sessionFactory, Statement statement) {
                return new PostgreSQLSqlAstTranslator<>(sessionFactory, statement) {
                    @Override
                    public void visitOffsetFetchClause(QueryPart queryPart) {
                        renderLimitOffsetClause(queryPart);
                    }
                };
            }
        };
    }
}
